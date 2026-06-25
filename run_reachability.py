"""
run_reachability.py
-------------------
End-to-end runner for Phase 3 reachability analysis on the dual_loop_run experiment.

Usage (from project root):
    python run_reachability.py                        # run one sample trajectory
    python run_reachability.py --n_samples 10         # run 10 trajectories
    python run_reachability.py --eval                 # also compute inclusion accuracy
    python run_reachability.py --show_plot            # show matplotlib plots interactively
"""

import argparse
import os
import pickle
import numpy as np
import logging

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load config (resolves all paths relative to project root)
# ---------------------------------------------------------------------------
from src.clustering.run import load_config
from src.reachability_analysis.simulation import run_scenario
from src.reachability_analysis.input_state import filter_paddings

# Phase 1+2 base experiment — clusters live here
CLUSTER_SOURCE = "dual_loop_run"
# Phase 2.5 fine-tuned model — intent head is active here (Loop B)
MODEL_FILE     = "finetune_intent"
FOLDER         = "experiments"
INDEX          = 2
INDEX_DATA     = 0


def main(n_samples: int = 1, show_plot: bool = False, run_eval: bool = False, save_plots: bool = True):
    logger.info("Loading config (fine-tuned model: finetune_intent) ...")
    config = load_config(
        folder=FOLDER,
        model_file=MODEL_FILE,
        index=INDEX,
        index_data=INDEX_DATA,
    )

    # Clusters were built on dual_loop_run — point there explicitly
    cluster_config = load_config(
        folder=FOLDER,
        model_file=CLUSTER_SOURCE,
        index=INDEX,
        index_data=INDEX_DATA,
    )
    clusters_dir = os.path.join(cluster_config["output_dir"], "clusters")
    logger.info(f"Clusters directory: {clusters_dir}")

    # Load the pre-computed cluster data (already on disk from Phase 2)
    logger.info("Loading cluster data from disk ...")
    with open(os.path.join(clusters_dir, "data_original.pkl"), "rb") as f:
        data_raw = pickle.load(f)
    with open(os.path.join(clusters_dir, "data_padding.pkl"), "rb") as f:
        paddings = pickle.load(f)
    with open(os.path.join(clusters_dir, "cluster_labels.pkl"), "rb") as f:
        labels_arr = pickle.load(f)

    # Filter out padded entries
    data_filtered   = filter_paddings(data_raw, paddings)
    labels_filtered = filter_paddings(labels_arr, paddings)

    # -----------------------------------------------------------------------
    # CRITICAL FIX: Denormalize data to physical units (meters, m/s, rad)
    # -----------------------------------------------------------------------
    # If the data is normalized between 0-1, the physical equations (Speed * cos(Heading))
    # break, causing the LTI pseudo-inverse M_sigma to explode and overflow to NaN.
    ROOT_PROJECT = os.path.abspath(os.path.dirname(__file__))
    exp_dir = os.path.join(ROOT_PROJECT, FOLDER, MODEL_FILE)
    norm_path = os.path.join(exp_dir, "norm_constants.npy")
    if not os.path.exists(norm_path):
        norm_path = os.path.join(os.path.join(ROOT_PROJECT, FOLDER, CLUSTER_SOURCE), "norm_constants.npy")
        if not os.path.exists(norm_path):
            norm_path = os.path.join(os.path.join(ROOT_PROJECT, FOLDER, CLUSTER_SOURCE), "eval", "norm_constants.npy")
    
    if os.path.exists(norm_path):
        logger.info(f"Denormalizing features to physical units using {norm_path}")
        norm = np.load(norm_path, allow_pickle=True).item()
        if "min_val" in norm:
            min_val = norm["min_val"]
            max_val = norm["max_val"]
            data_filtered = data_filtered * ((max_val - min_val) + 1e-8) + min_val
        elif "mean" in norm:
            mean = norm["mean"]
            std = norm["std"]
            data_filtered = data_filtered * (std + 1e-8) + mean
    else:
        logger.warning("No norm_constants.npy found! Reachability math may overflow if data is normalized.")

    # Remove HDBSCAN noise points (label == -1) — they have no meaningful cluster
    noise_mask      = labels_filtered != -1
    data_filtered   = data_filtered[noise_mask]
    labels_filtered = labels_filtered[noise_mask]

    cluster_ids = sorted(set(int(l) for l in labels_filtered))  # e.g. [0,1,...,94]
    n_clusters  = max(cluster_ids) + 1   # size for range() in separate_data_to_class
    logger.info(f"Found {len(cluster_ids)} clusters (noise removed): {cluster_ids[:10]}...")
    logger.info(f"Total trajectory chunks available (after noise removal): {len(data_filtered)}")

    # Filter out stationary trajectories (speed=0 for all timesteps).
    # ~91% of the data are parked cars with zero dynamics, which cause the LTI
    # system identification to produce degenerate (all-zero) matrices.
    from src.reachability_analysis.input_state import IDX_SPEED
    moving_mask = data_filtered[:, :, IDX_SPEED].max(axis=1) > 0.001
    moving_indices = np.where(moving_mask)[0]
    logger.info(f"Moving trajectories (speed > 0): {len(moving_indices)} / {len(data_filtered)}")

    # Pick n_samples random trajectory chunks from MOVING vehicles only
    rng     = np.random.default_rng(seed=42)
    sampled = rng.choice(moving_indices, size=min(n_samples, len(moving_indices)), replace=False)
    indices = sampled

    reachable_sets_all = {}
    ground_truth_all   = {}

    os.makedirs(os.path.join(config["output_dir"], "VANET"), exist_ok=True)

    for run_idx, traj_idx in enumerate(indices):
        trajectory = data_filtered[traj_idx]          # shape: (seq_len, feat_dim)
        true_label = int(labels_filtered[traj_idx])
        logger.info(f"\n[{run_idx+1}/{len(indices)}] trajectory index={traj_idx} | cluster={true_label}")

        save_path = None
        if save_plots:
            save_path = os.path.join(config["output_dir"], "VANET", f"reachability_traj_{run_idx}")

        from src.reachability_analysis.simulation import (
            get_initial_conditions,
            reachability_for_specific_cluster,
        )
        from src.reachability_analysis.input_state import separate_data_to_class, structure_input_data_for_clusters

        # Cap each cluster's sample size to prevent massive RAM spikes and extremely long evaluation times
        # Use only moving trajectories for system identification (stationary = all-zero matrices)
        data_moving = data_filtered[moving_mask]
        labels_moving = labels_filtered[moving_mask]
        data_capped, labels_capped = structure_input_data_for_clusters(data_moving, labels_moving, max_data=50)
        data_by_cluster = separate_data_to_class(data_capped, labels_capped, size=n_clusters)

        pos, vel = get_initial_conditions(trajectory)
        real_pos = pos.copy()
        pos_origin = np.array([0.0, 0.0])
        logger.info(f"  pos={real_pos},  vel={vel}")

        # Compute reachability time-series specifically for the True Cluster to evaluate inclusion accuracy
        # Compute at origin [0,0] to use Shift-Invariant kinematics, then shift back to global coordinates
        result = reachability_for_specific_cluster(
            pos=pos_origin,
            vel=vel,
            cluster_id=true_label,
            data=data_by_cluster,
            config=config,
            baseline=True,
            show_plot=False,
            ax=None,
            suppress_prints=False,
            method=str(true_label)
        )

        if result is None:
            logger.warning("  Reachability failed (no data/constraints matched).")
            continue
            
        ax, zonos, R_all, R_base_all = result

        # Shift all calculated zonotopes back to the vehicle's true global position
        for z in zonos:
            z.x = z.x + real_pos.reshape(2, 1)
        for z in R_all:
            z.x = z.x + real_pos.reshape(2, 1)
        if R_base_all is not None:
            for z in R_base_all:
                z.x = z.x + real_pos.reshape(2, 1)

        if show_plot or save_path:
            from src.reachability_analysis.operations import visualize_zonotopes
            try:
                visualize_zonotopes(zonos, save_plot=save_path, show=show_plot, title=f"Trajectory {run_idx+1} | True cluster: {true_label}")
            except Exception as e:
                logger.warning(f"  Could not visualize zonotopes (degenerate shape): {e}")

        # Store for inclusion accuracy evaluation
        # ground truth = future X,Y positions from this trajectory chunk
        from src.reachability_analysis.input_state import IDX_X, IDX_Y
        gt_xy = trajectory[:, [IDX_X, IDX_Y]]  # (seq_len, 2)
        ground_truth_all[run_idx] = gt_xy
        
        # Store the TIME SERIES reachable sets for inclusion evaluation
        reachable_sets_all[run_idx] = R_all

    # Save pickles for inclusion accuracy
    vanet_dir = os.path.join(config["output_dir"], "VANET")
    with open(os.path.join(vanet_dir, "reachable_sets.pkl"), "wb") as f:
        pickle.dump(reachable_sets_all, f)
    with open(os.path.join(vanet_dir, "ground_truth.pkl"), "wb") as f:
        pickle.dump(ground_truth_all, f)
    logger.info(f"Saved reachable sets and ground truth to {vanet_dir}/")

    # Optional: run inclusion accuracy evaluation
    if run_eval:
        logger.info("\nRunning inclusion accuracy evaluation ...")
        from src.reachability_analysis.inclusion_accuracy import get_state_inclusion_acc
        results = get_state_inclusion_acc(
            config,
            save_path=os.path.join(vanet_dir, "accuracy.png"),
        )
        logger.info(f"Peak accuracy: {results['accuracy_per_step'].max():.1f}%")

    logger.info("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 vehicle reachability analysis.")
    parser.add_argument("--n_samples",  type=int,  default=1,
                        help="Number of trajectory chunks to run reachability on (default: 1)")
    parser.add_argument("--show_plot",  action="store_true",
                        help="Show matplotlib plots interactively")
    parser.add_argument("--no_save",    action="store_true",
                        help="Do not save plots to disk")
    parser.add_argument("--eval",       action="store_true",
                        help="Run inclusion accuracy evaluation after reachability")
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        show_plot=args.show_plot,
        run_eval=args.eval,
        save_plots=not args.no_save,
    )
