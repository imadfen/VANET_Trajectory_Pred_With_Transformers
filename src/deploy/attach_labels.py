"""
src/deploy/attach_labels.py
===========================
Step 2.5 — Generate cluster-derived intent labels and attach them to the
trainer so that `main.py` can fine-tune the Intent Head.

What this script does:
    1. Loads `output_data.pt` from the Phase 1 experiment (embeddings + chunk IDs).
    2. Loads the HDBSCAN cluster assignments from Phase 2 (`clusters/` folder).
    3. Maps each HDBSCAN cluster → one of the 4 vehicle intent labels
       (MaintainLane=0, Turn=1, Exit=2, Brake=3).
    4. Writes a `intent_labels.pt` file that contains a long tensor of shape
       (N_chunks,) where each entry is an integer label 0-3.
    5. Updates `configuration.json` in the experiment folder to set
       `intent_weight = 1.0` so the Intent Head is trained on the next run.

After running this script:
    python main.py \\
        --config=experiments/<model_file>/configuration.json \\
        --load_model=experiments/<model_file>/checkpoints/model_best.pth \\
        --name=finetune_intent

The model runner automatically picks up `intent_labels.pt` if it lives
alongside `output_data.pt` in the eval output directory.

Usage:
    python src/deploy/attach_labels.py \\
        --folder=experiments \\
        --model_file=dual_loop_run_2026-XX-XX_XX-XX-XX_XXX \\
        --intent_weight=1.0
"""

import sys
import os
import json
import argparse
import logging

import numpy as np
import torch

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Intent label constants (must match Labels.py and loop_b.py) ───────────────
VEHICLE_LABELS = {
    0: "MaintainLane",
    1: "Turn",
    2: "Exit",
    3: "Brake",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_output_data(output_dir: str):
    """Load Phase 1 output_data.pt."""
    pt_path = os.path.join(output_dir, "output_data.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"output_data.pt not found at: {pt_path}\n"
            "Run Phase 1 (main.py with eval_only) first."
        )
    data = torch.load(pt_path, map_location="cpu")
    logger.info(f"Loaded output_data.pt from: {pt_path}")
    return data


def load_cluster_assignments(clusters_dir: str):
    """Load HDBSCAN cluster label array saved by Phase 2 run.py.

    Returns np.ndarray of shape (N_chunks,) with cluster IDs (-1 = noise).
    """
    import pickle
    label_path = os.path.join(clusters_dir, "cluster_labels.pkl")
    
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"cluster_labels.pkl not found at: {clusters_dir}\n"
            "Run Phase 2 (src/clustering/run.py) first."
        )
            
    with open(label_path, "rb") as f:
        labels = pickle.load(f)
        
    logger.info(f"Loaded cluster labels: {labels.shape}, unique={np.unique(labels)}")
    return labels


def hdbscan_to_intent(cluster_labels: np.ndarray, output_data: dict, exp_dir: str) -> np.ndarray:
    """Map raw HDBSCAN integer cluster IDs to the 4 intent classes based on physical features.

    Strategy:
      - Turn (1): Absolute AngularVelocity > 0.05 rad/s
      - Brake (3): Acceleration < -1.0 m/s^2
      - Exit (2): Absolute LaneDist > 1.0 m (and not turning)
      - MaintainLane (0): All remaining clusters

    Returns np.ndarray of shape (N,) with values in {0, 1, 2, 3}.
    Noise samples (cluster_id == -1) are assigned label 0 (MaintainLane).
    """
    unique_clusters = sorted(set(cluster_labels.tolist()) - {-1})
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        raise RuntimeError(
            "No valid clusters found (all noise). "
            "Try reducing min_cluster_size or min_samples in Phase 2."
        )

    logger.info(f"Mapping {n_clusters} HDBSCAN clusters → 4 intent classes using physical thresholds")

    # 1. Extract and pool physical features
    targets = output_data["targets"]
    padding_masks = output_data["padding_masks"]
    
    if hasattr(targets, "numpy"):
        targets = targets.numpy()
    if hasattr(padding_masks, "numpy"):
        padding_masks = padding_masks.numpy()
        
    # Pool features by taking the mean over the time dimension for valid timesteps
    pooled_features = np.ma.masked_array(
        targets, mask=np.broadcast_to(~padding_masks[:, :, None], targets.shape)
    ).mean(axis=1).data
    
    # 2. Denormalize the data back to physical units
    norm_path = os.path.join(exp_dir, "norm_constants.npy")
    if not os.path.exists(norm_path):
        norm_path = os.path.join(exp_dir, "eval", "norm_constants.npy")
        
    if os.path.exists(norm_path):
        logger.info(f"Denormalizing features using {norm_path}")
        norm = np.load(norm_path, allow_pickle=True).item()
        if "min_val" in norm:
            min_val = norm["min_val"]
            max_val = norm["max_val"]
            pooled_features = pooled_features * ((max_val - min_val) + 1e-8) + min_val
        elif "mean" in norm:
            mean = norm["mean"]
            std = norm["std"]
            pooled_features = pooled_features * (std + 1e-8) + mean
    else:
        logger.warning(f"No norm_constants.npy found in {exp_dir}. Using raw values, which may be scaled!")
    
    # 3. Calculate median physical stats for each cluster
    cluster_stats = {}
    for cid in unique_clusters:
        idx = (cluster_labels == cid)
        cluster_data = pooled_features[idx]
        
        # 3: Acceleration, 5: AngularVelocity, 7: LaneDist
        med_accel = np.median(cluster_data[:, 3])
        med_abs_ang_vel = np.median(np.abs(cluster_data[:, 5]))
        med_abs_lane_dist = np.median(np.abs(cluster_data[:, 7]))
        
        cluster_stats[cid] = {
            "accel": med_accel,
            "abs_ang_vel": med_abs_ang_vel,
            "abs_lane_dist": med_abs_lane_dist
        }
        logger.info(f"  Cluster {cid:3d} Stats -> Accel: {med_accel:.4f}, AbsAngVel: {med_abs_ang_vel:.4f}, AbsLaneDist: {med_abs_lane_dist:.4f}")

    # 4. Heuristic Assignment (Physical Thresholds)
    cluster_to_intent = {}
    for cid in unique_clusters:
        stats = cluster_stats[cid]
        accel = stats["accel"]
        abs_ang_vel = stats["abs_ang_vel"]
        abs_lane_dist = stats["abs_lane_dist"]
        
        # Assign intent based on physics (priority ordered for safety)
        if abs_ang_vel > 0.05:          # Significant angular rotation -> Turn
            intent = 1
        elif accel < -1.0:              # Hard deceleration -> Brake
            intent = 3
        elif abs_lane_dist > 1.0:       # Lateral displacement without extreme turn -> Exit/Lane change
            intent = 2
        else:                           # Default stable cruising -> MaintainLane
            intent = 0
            
        cluster_to_intent[cid] = intent

    logger.info("Cluster → Intent Final Mapping:")
    for cid in unique_clusters:
        intent = cluster_to_intent[cid]
        logger.info(f"  HDBSCAN cluster {cid:3d}  →  {intent} ({VEHICLE_LABELS[intent]})")

    intent_labels = np.zeros(len(cluster_labels), dtype=np.int64)
    for idx, cid in enumerate(cluster_labels):
        if cid == -1:
            intent_labels[idx] = 0   # noise → MaintainLane
        else:
            intent_labels[idx] = cluster_to_intent[cid]

    # Distribution report
    unique, counts = np.unique(intent_labels, return_counts=True)
    logger.info("Intent label distribution:")
    for u, c in zip(unique, counts):
        pct = 100.0 * c / len(intent_labels)
        logger.info(f"  {VEHICLE_LABELS[u]:12s} ({u})  →  {c:6d} chunks  ({pct:.1f}%)")

    return intent_labels


def save_intent_labels(intent_labels: np.ndarray, output_data: dict, output_dir: str) -> str:
    """Save intent_labels.pt mapped explicitly via global IDs."""
    out_path = os.path.join(output_dir, "intent_labels.pt")
    
    # 1. Get raw dataset IDs for each labeled chunk
    chunk_ids = output_data["IDs"]
    if hasattr(chunk_ids, "numpy"):
        chunk_ids = chunk_ids.numpy()
        
    # 2. Build a globally spanning sparse tensor defaulting to -100
    # PyTorch's F.cross_entropy defaults to ignore_index=-100 natively.
    # Therefore, any index without a label inherently skips gradient backprop without punishing the network!
    max_id = max(chunk_ids) if len(chunk_ids) > 0 else 0
    safe_tensor_size = max(1000000, max_id + 500000)
    
    sparse_intent_labels = torch.full((safe_tensor_size,), -100, dtype=torch.long)
    sparse_intent_labels[chunk_ids] = torch.tensor(intent_labels, dtype=torch.long)
    
    torch.save(sparse_intent_labels, out_path)
    
    # Clean up heavy sparse tensors instantly
    del sparse_intent_labels, chunk_ids
    
    logger.info(f"Saved sparse global intent labels mapped to IDs → {out_path}")
    return out_path


def patch_config(config_path: str, intent_weight: float, labels_path: str):
    """Update configuration.json to activate intent fine-tuning."""
    with open(config_path) as f:
        config = json.load(f)

    config["intent_weight"] = intent_weight
    config["intent_labels_path"] = labels_path   # new key for model runner

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(
        f"Updated configuration.json:\n"
        f"  intent_weight       = {intent_weight}\n"
        f"  intent_labels_path  = {labels_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def attach_labels(
    folder:       str,
    model_file:   str,
    intent_weight: float = 1.0,
):
    exp_dir     = os.path.join(ROOT, folder, model_file)
    output_dir  = os.path.join(exp_dir, "eval")
    clusters_dir = os.path.join(output_dir, "clusters")
    config_path  = os.path.join(exp_dir, "configuration.json")

    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"configuration.json not found at: {config_path}")

    # 1. Load Phase 1 output to verify chunk count
    output_data   = load_output_data(output_dir)
    
    if isinstance(output_data["embeddings"], list):
        n_chunks = sum(e.shape[0] for e in output_data["embeddings"])
    else:
        n_chunks = output_data["embeddings"].shape[0]
        
    logger.info(f"Total chunks in output_data.pt: {n_chunks}")

    # 2. Load Phase 2 cluster assignments
    cluster_labels = load_cluster_assignments(clusters_dir)

    if len(cluster_labels) != n_chunks:
        raise ValueError(
            f"Mismatch: output_data.pt has {n_chunks} chunks, "
            f"but cluster_labels.npy has {len(cluster_labels)} entries.\n"
            "Re-run Phase 2 on the same experiment folder."
        )

    # 3. Map HDBSCAN clusters → intent labels (0-3)
    intent_labels = hdbscan_to_intent(cluster_labels, output_data, exp_dir)

    # 4. Save intent_labels.pt
    labels_path = save_intent_labels(intent_labels, output_data, output_dir)
    
    # 5. Clear aggressive memory blocks explicitly 
    del output_data, cluster_labels

    # 6. Patch configuration.json
    patch_config(config_path, intent_weight, labels_path)

    logger.info("✅Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2.5: attach cluster-derived intent labels to the "
                    "experiment so main.py can fine-tune the Intent Head."
    )
    parser.add_argument("--folder",       type=str, default="experiments",
                        help="Top-level experiments folder")
    parser.add_argument("--model_file",   type=str, required=True,
                        help="Experiment subfolder (Phase 1 output)")
    parser.add_argument("--intent_weight", type=float, default=1.0,
                        help="Weight for intent cross-entropy loss (default 1.0)")

    args = parser.parse_args()

    attach_labels(
        folder        = args.folder,
        model_file    = args.model_file,
        intent_weight = args.intent_weight,
    )
