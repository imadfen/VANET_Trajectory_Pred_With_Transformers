"""
src/deploy/export_decisions.py
==============================
Offline batch script — generates a `decisions.json` file that maps every
vehicle + timestep to the Loop A / Loop B decisions produced by the trained
Transformer model.

This file is consumed by `DataCollectorApp.cc` in the second OMNeT++ run
to apply beacon suppression (Loop A) and MAC backoff biasing (Loop B)
without any live socket communication.

Usage (from project root):
    python src/deploy/export_decisions.py \\
        --folder=experiments \\
        --model_file=dual_loop_run_2026-05-17_XX-XX-XX_XXX \\
        --data_dir=resources/VANET_data/raw/dataset-35m-10hz-packet_loss_32%/raw/ \\
        --pattern=data_car_ \\
        --output=decisions.json

Output format (decisions.json):
    {
        "car_32": {
            "18003.10": {"beacon_hz": 2.0,  "mac_wait_ms": 100.0, "flag": 0, "intent": "MaintainLane"},
            "18003.20": {"beacon_hz": 10.0, "mac_wait_ms": 1.0,   "flag": 1, "intent": "Brake"},
            ...
        },
        ...
    }
"""

import sys
import os
import json
import argparse
import glob
import logging

import numpy as np
import torch

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from src.loops.loop_a import DiscrepancyMonitor
from src.loops.loop_b import StabilityScorer, MACBiasMapper
from src.clustering.run import load_config, load_embeddings_from_pt

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# ── Feature layout (must match VANET_FEATURE_NAMES) ──────────────────────────
IDX_TIME = None   # 'Time' column — read separately, not a model feature
N_FEATURES = 51   # columns after dropping 'Time'


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_vehicle_csv(filepath: str):
    """Read a single per-vehicle CSV and return (times, features).

    Returns
    -------
    times    : np.ndarray  shape (T,)   — simulation timestamps (seconds)
    features : np.ndarray  shape (T, 51) — feature matrix, Time column removed
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    times    = df["Time"].values
    features = df.drop(columns=["Time"]).values.astype(np.float32)
    return times, features


def sliding_windows(features: np.ndarray, seq_len: int, step: int = 1):
    """Yield (start_idx, window) pairs from a feature matrix.

    Parameters
    ----------
    features : (T, F)
    seq_len  : number of timesteps per window
    step     : stride between consecutive windows
    """
    T = features.shape[0]
    for i in range(0, T - seq_len + 1, step):
        yield i, features[i : i + seq_len]


# ─────────────────────────────────────────────────────────────────────────────
# Model inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(config: dict, device: torch.device):
    """Load the pre-trained Transformer from the experiment checkpoint."""
    from src.transformer_model.model import create_model

    model = create_model(config, device)
    ckpt_path = config["load_model"]
    logger.info(f"Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_window(model, window: np.ndarray, device: torch.device):
    """Run one (seq_len, 51) window through the Transformer.

    Returns
    -------
    prediction    : np.ndarray  (seq_len, 51)  — reconstructed features
    intent_logits : np.ndarray  (4,)           — raw intent head output
                                                 (zeros if intent head inactive)
    """
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, F)
    padding_mask = torch.ones(1, window.shape[0], dtype=torch.bool).to(device)

    with torch.no_grad():
        outputs = model(x, padding_mask)

    # outputs is a tuple: (reconstruction, intent_logits, attn_maps, embeddings)
    recon = outputs[0].squeeze(0).cpu().numpy()   # (T, F)

    intent_logits = np.zeros(4, dtype=np.float32)
    if outputs[1] is not None:
        intent_logits = outputs[1].squeeze(0).cpu().numpy()  # (4,)

    return recon, intent_logits


# ─────────────────────────────────────────────────────────────────────────────
# Main export routine
# ─────────────────────────────────────────────────────────────────────────────

def export_decisions(
    folder:     str,
    model_file: str,
    data_dir:   str,
    pattern:    str = "data_car_",
    output:     str = "decisions.json",
    seq_len:    int = 60,
    step:       int = 1,
    epsilon:    float = 0.5,
    min_wait_ms: float = 1.0,
    max_wait_ms: float = 100.0,
):
    # ── Load config and model ─────────────────────────────────────────────────
    config = load_config(folder=folder, model_file=model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on: {device}")

    model = load_model(config, device)

    # ── Initialise Loop A / B ─────────────────────────────────────────────────
    monitor = DiscrepancyMonitor(
        epsilon        = epsilon,
        beacon_hz_low  = config.get("beacon_hz_low",  2.0),
        beacon_hz_high = config.get("beacon_hz_high", 10.0),
    )
    scorer = StabilityScorer()
    mapper = MACBiasMapper(min_wait_ms=min_wait_ms, max_wait_ms=max_wait_ms)

    # ── Find all vehicle CSV files ────────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(data_dir, f"{pattern}*.csv")))
    logger.info(f"Found {len(csv_files)} vehicle files in: {data_dir}")

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{pattern}*.csv' found in {data_dir}"
        )

    # ── Process each vehicle ──────────────────────────────────────────────────
    all_decisions = {}
    prev_prediction = None

    for filepath in csv_files:
        car_id = os.path.splitext(os.path.basename(filepath))[0]  # e.g. data_car_32_t18003
        logger.info(f"Processing: {car_id}")

        times, features = load_vehicle_csv(filepath)

        if features.shape[0] < seq_len:
            logger.warning(f"  Skipping {car_id} — only {features.shape[0]} rows < seq_len={seq_len}")
            continue

        if features.shape[1] != N_FEATURES:
            logger.warning(
                f"  Skipping {car_id} — expected {N_FEATURES} features, got {features.shape[1]}"
            )
            continue

        car_decisions = {}
        prev_prediction = None

        for start_idx, window in sliding_windows(features, seq_len, step):
            # ── Model inference ───────────────────────────────────────────────
            prediction, intent_logits = infer_window(model, window, device)

            # ── Loop A: compare previous prediction to current actual ─────────
            if prev_prediction is not None:
                actual_window = features[start_idx : start_idx + seq_len]
                loop_a = monitor.check(prev_prediction, actual_window)
            else:
                # First window — no prior prediction available, default to low entropy
                from src.loops.loop_a import LoopADecision
                loop_a = LoopADecision(flag=0, residual=0.0,
                                       beacon_hz=monitor.beacon_hz_low)

            # ── Loop B: stability scoring on intent logits ────────────────────
            logit_tensor = torch.tensor(intent_logits)
            loop_b       = scorer.score(logit_tensor)
            mac_wait_ms  = mapper.map(loop_b.P_stable)

            # ── Record decision keyed to the LAST timestep of the window ──────
            ts_key = f"{times[start_idx + seq_len - 1]:.2f}"
            car_decisions[ts_key] = {
                "flag":        int(loop_a.flag),
                "mac_wait_ms": round(float(mac_wait_ms), 2),
            }

            prev_prediction = prediction

        all_decisions[car_id] = car_decisions
        logger.info(f"  → {len(car_decisions)} decisions recorded")

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = os.path.join(ROOT, output)
    with open(out_path, "w") as f:
        json.dump(all_decisions, f, indent=2)

    logger.info(f"\nDecision file written to: {out_path}")
    logger.info(f"  Total vehicles : {len(all_decisions)}")
    logger.info(f"  Total decisions: {sum(len(v) for v in all_decisions.values())}")

    # ── Quick summary stats ───────────────────────────────────────────────────
    all_hz = [
        d["beacon_hz"]
        for v in all_decisions.values()
        for d in v.values()
    ]
    suppressed = sum(1 for hz in all_hz if hz <= 2.0)
    logger.info(
        f"\nLoop A summary: {suppressed}/{len(all_hz)} "
        f"({100*suppressed/max(len(all_hz),1):.1f}%) timesteps suppressed to 2 Hz"
    )

    intents = [
        d["intent"]
        for v in all_decisions.values()
        for d in v.values()
    ]
    from collections import Counter
    logger.info(f"Loop B intent distribution: {dict(Counter(intents))}")

    return all_decisions


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export per-vehicle Loop A/B decisions to decisions.json "
                    "for deterministic replay in OMNeT++."
    )
    parser.add_argument("--folder",      type=str, default="experiments",
                        help="Top-level experiments folder")
    parser.add_argument("--model_file",  type=str, required=True,
                        help="Experiment subfolder name (e.g. dual_loop_run_2026-…)")
    parser.add_argument("--data_dir",    type=str,
                        default="resources/VANET_data/raw/dataset-35m-10hz-packet_loss_32%/raw/",
                        help="Directory containing per-vehicle CSV files")
    parser.add_argument("--pattern",     type=str, default="data_car_",
                        help="CSV filename prefix")
    parser.add_argument("--output",      type=str, default="decisions.json",
                        help="Output file path (relative to project root)")
    parser.add_argument("--seq_len",     type=int, default=60,
                        help="Sliding window length (timesteps)")
    parser.add_argument("--step",        type=int, default=1,
                        help="Stride between consecutive windows")
    parser.add_argument("--epsilon",     type=float, default=0.5,
                        help="Loop A discrepancy threshold")
    parser.add_argument("--min_wait_ms", type=float, default=1.0,
                        help="Loop B minimum MAC backoff (ms)")
    parser.add_argument("--max_wait_ms", type=float, default=100.0,
                        help="Loop B maximum MAC backoff (ms)")

    args = parser.parse_args()

    export_decisions(
        folder      = args.folder,
        model_file  = args.model_file,
        data_dir    = args.data_dir,
        pattern     = args.pattern,
        output      = args.output,
        seq_len     = args.seq_len,
        step        = args.step,
        epsilon     = args.epsilon,
        min_wait_ms = args.min_wait_ms,
        max_wait_ms = args.max_wait_ms,
    )
