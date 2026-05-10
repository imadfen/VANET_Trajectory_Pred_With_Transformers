"""
Phase 3 — Vehicle Reachability Simulation

Replaces pedestrian kinematic limits (0-2 m/s) with vehicle dynamics:
  - Speed range:            0–50 m/s  (highway), 0–20 m/s (urban)
  - Max lateral accel:      ±0.3 g  (~3 m/s²)
  - Heading change rate:    bounded by minimum turn radius at current speed

Entry points:
  reachability_for_specific_cluster(pos, vel, cluster_id, data, config, ...)
  run_scenario(trajectory, config, cluster_labels, ...)
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle
import json
import argparse
import logging

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_dir)

from src.reachability_analysis.operations import (
    visualize_zonotopes,
    input_zonotope,
    create_M_w,
    zonotope_area,
)
from src.reachability_analysis.reachability import LTI_reachability
from src.reachability_analysis.input_state import (
    create_io_state,
    separate_data_to_class,
    split_io_to_trajs,
    filter_paddings,
    VEHICLE_LABELS,
    IDX_X, IDX_Y, IDX_SPEED, IDX_HEADING,
)
from src.reachability_analysis.zonotope import zonotope
from src.reachability_analysis.utils import load_data
from src.clustering.run import get_cluster, load_config

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_PROJECT = os.getcwd()
ROOT_RESOURCES = os.path.join(ROOT_PROJECT, "resources")

# ---------------------------------------------------------------------------
# Vehicle kinematic constants
# ---------------------------------------------------------------------------
MAX_SPEED_HIGHWAY = 50.0      # m/s
MAX_SPEED_URBAN   = 20.0      # m/s
MAX_LAT_ACCEL     = 3.0       # m/s²  (≈ 0.3 g)
MIN_TURN_RADIUS   = 5.0       # m  (sharp urban corner)
PROCESS_NOISE_STD = 0.1       # m   (GPS + model noise)

COLORS = [
    [60 / 255, 159 / 255, 69 / 255,  0.4],   # Behavioral Zonotope
    [32 / 255, 102 / 255, 168 / 255, 0.4],   # Transformer-cluster Zonotope
    [0.55, 0.14, 0.14, 0.4],                 # Baseline Zonotope
    [0.25, 0.25, 0.25, 0.6],                 # Current Position
    [63  / 255, 63  / 255, 63  / 255, 1.0],  # Past Trajectory
    [184 / 255, 184 / 255, 184 / 255, 1.0],  # Future Trajectory
]

REVERSED_LABELS = {v: k for k, v in VEHICLE_LABELS.items()}


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def get_initial_conditions(traj_chunk: np.ndarray):
    """Extract (pos, vel) from the last row of a (seq_len, 23) VANET chunk."""
    last = traj_chunk[-1]
    pos = np.array([last[IDX_X], last[IDX_Y]])
    speed   = float(last[IDX_SPEED])
    heading = float(last[IDX_HEADING])
    vel = np.array([speed * np.cos(heading), speed * np.sin(heading)])
    return pos, vel


# ---------------------------------------------------------------------------
# Single-cluster reachability
# ---------------------------------------------------------------------------

def reachability_for_specific_cluster(
    pos: np.ndarray,
    vel: np.ndarray,
    cluster_id: int,
    data: dict,
    config: dict,
    baseline: bool = True,
    show_plot: bool = False,
    ax: plt.Axes = None,
    suppress_prints: bool = True,
    method: str = "",
    data_statistics: dict = None,
    clustering: bool = False,
):
    """Compute reachable set for a vehicle at (pos, vel) belonging to cluster_id.

    Parameters
    ----------
    pos : np.ndarray  shape (2,)  – current (X, Y) in metres
    vel : np.ndarray  shape (2,)  – current velocity (vx, vy) in m/s
    cluster_id : int
    data : dict  – output of separate_data_to_class(...)
    config : dict
    """
    input_len = np.array(list(data.values())[0]).shape[1]
    N_reach = input_len - 1

    # Vehicle-appropriate noise and initial set
    z_w = zonotope(np.zeros(2), PROCESS_NOISE_STD * np.ones((2, 1)))
    G_z_init = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.6]])
    z = zonotope(pos, G_z_init)

    res = create_io_state(
        data, z, vel, cluster_id,
        drop_equal=False, angle_filter=True,
        method=method, data_statistics=data_statistics,
        clustering=clustering,
    )
    if res is None:
        return None

    U, X_p, X_m, _ = res
    _, _, U_traj = split_io_to_trajs(X_p, X_m, U, threshold=5.0, dropped=False, N=N_reach)
    U_k = input_zonotope(U_traj, N=N_reach)
    M_w = create_M_w(U.shape[1], z_w, disable_progress_bar=suppress_prints)

    # Tighter initial set for final propagation
    G_z_tight = np.array([[0.5, 0.0, 0.25], [0.0, 0.5, 0.15]])
    z = zonotope(pos, G_z_tight)
    R_all = LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k,
                             N=N_reach, disable_progress_bar=suppress_prints)
    R = R_all[-1]
    R.color = COLORS[0]

    # Baseline (all clusters combined, no angle filter)
    R_base_all = None
    R_base = None
    if baseline:
        z_base = zonotope(pos, G_z_tight)
        res_b = create_io_state(
            data, z_base, vel,
            list(data.keys()),        # all clusters
            drop_equal=False, angle_filter=False,
            method="baseline", data_statistics=data_statistics,
            clustering=False,
        )
        if res_b is not None:
            U_b, X_p_b, X_m_b, _ = res_b
            _, _, U_traj_b = split_io_to_trajs(X_p_b, X_m_b, U_b, threshold=5.0, dropped=False, N=N_reach)
            U_k_b = input_zonotope(U_traj_b, N=N_reach)
            M_w_b = create_M_w(U_b.shape[1], z_w, disable_progress_bar=suppress_prints)
            z_base2 = zonotope(pos, G_z_tight)
            R_base_all = LTI_reachability(U_b, X_p_b, X_m_b, z_base2, z_w, M_w_b, U_k_b,
                                          N=N_reach, disable_progress_bar=suppress_prints)
            R_base = R_base_all[-1]
            R_base.color = COLORS[2]

    if not suppress_prints:
        logger.info(f"Reachable set area: {zonotope_area(R):.4f} m²")
        if R_base is not None:
            logger.info(f"Baseline area: {zonotope_area(R_base):.4f} m²")

    z.color = COLORS[3]
    zonos = [R_base, R, z] if (baseline and R_base) else [R, z]

    if ax and show_plot:
        visualize_zonotopes(zonos, map=ax, show=show_plot)

    return ax, zonos, R_all, R_base_all


# ---------------------------------------------------------------------------
# Multi-cluster scenario (all clusters for a single vehicle state)
# ---------------------------------------------------------------------------

def reachability_for_all_clusters(
    pos: np.ndarray,
    vel: np.ndarray,
    config: dict,
    data: dict,
    cluster_ids: list,
    baseline: bool = True,
    show_plot: bool = False,
    save_plot: str = None,
    data_statistics: dict = None,
    title: str = "",
):
    """Run reachability for each cluster label and overlay on one plot."""
    _z, _labels = [], []
    _baseline_zonotopes = []
    ax = None

    for cluster_id in cluster_ids:
        method = VEHICLE_LABELS.get(cluster_id, str(cluster_id))
        if data_statistics is not None:
            data_statistics.setdefault("total_count", 0)
            data_statistics["total_count"] += 1

        result = reachability_for_specific_cluster(
            pos, vel, cluster_id, data, config,
            baseline=(baseline and not _baseline_zonotopes),
            ax=ax, suppress_prints=True,
            method=method, data_statistics=data_statistics,
        )
        if result is None:
            continue

        ax, zonos, R_all, R_base_all = result
        R = zonos[-2]  # second-to-last is the cluster zonotope
        R.color = COLORS[cluster_id % len(COLORS)]
        _z.append(R)
        _labels.append(method)

        if R_base_all and not _baseline_zonotopes:
            _baseline_zonotopes.append(R_base_all[-1])

    if _baseline_zonotopes:
        _labels.insert(0, "Baseline")
        _z.insert(0, _baseline_zonotopes[0])

    visualize_zonotopes(_z, save_plot=save_plot, show=show_plot, _labels=_labels, title=title)
    return _z, _labels


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    trajectory: np.ndarray,   # (seq_len, 23) — current vehicle window
    config: dict,
    cluster_labels: list,
    show_plot: bool = False,
    baseline: bool = True,
    save_plot: str = None,
    title: str = "",
):
    """Top-level entry: extract initial conditions and run reachability."""
    pos, vel = get_initial_conditions(trajectory)

    # Load cluster data from saved embeddings
    from src.reachability_analysis.utils import load_data as _load
    data_raw = _load(filename="/data_original.pkl",
                     filepath=os.path.join(config["output_dir"], "clusters"))
    paddings = _load(filename="/data_padding.pkl",
                     filepath=os.path.join(config["output_dir"], "clusters"))
    labels_arr = _load(filename="/cluster_labels.pkl",
                       filepath=os.path.join(config["output_dir"], "clusters"))

    data_filtered = filter_paddings(data_raw, paddings)
    labels_filtered = filter_paddings(labels_arr, paddings)
    n_clusters = len(set(labels_filtered))
    data_by_cluster = separate_data_to_class(data_filtered, labels_filtered, size=n_clusters)

    reachability_for_all_clusters(
        pos, vel, config, data_by_cluster, cluster_labels,
        baseline=baseline, show_plot=show_plot, save_plot=save_plot, title=title,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vehicle reachability simulation (Phase 3).")
    parser.add_argument("--folder", type=str, default="experiments")
    parser.add_argument("--model_file", type=str, default="VANETDataset_pretrained")
    parser.add_argument("--index", type=int, default=2)
    parser.add_argument("--index_data", type=int, default=0)
    args = parser.parse_args()

    config = load_config(
        folder=args.folder,
        model_file=args.model_file,
        index=args.index,
        index_data=args.index_data,
    )
    logger.info("Reachability simulation ready. Call run_scenario() with a vehicle trajectory chunk.")
