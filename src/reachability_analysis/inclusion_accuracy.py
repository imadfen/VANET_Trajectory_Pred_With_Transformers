
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import argparse


REVERSED_LABELS = {}  # filled at import from VEHICLE_LABELS when needed
DATASET   = "vanet_test"      # update to your vehicle test dataset folder name
RA_PATH   = "/VANET/reachable_sets.pkl"
RAB_PATH  = "/VANET/reachable_base_sets.pkl"


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def compute_inclusion_accuracy(
    reachable_sets: dict,
    ground_truth: dict,
    max_area: float = 5000.0,
) -> dict:
    """Compute what fraction of actual vehicle positions fall inside predicted reachable sets.

    Parameters
    ----------
    reachable_sets : dict { vehicle_id -> list[pp.zonotope] }
        One zonotope per prediction timestep per vehicle.
    ground_truth : dict { vehicle_id -> np.ndarray (T, 2) }
        True (X, Y) positions in metres.
    max_area : float
        Zonotopes larger than this (m²) are degenerate and excluded.

    Returns
    -------
    dict with keys:
        'accuracy_per_step' : np.ndarray – % inside at each timestep
        'volume_per_step'   : np.ndarray – mean zonotope area (m²) at each timestep
        'n_evaluated'       : int        – max evaluations across all timesteps
    """
    from src.reachability_analysis.operations import is_inside, zonotope_area

    if not reachable_sets:
        return {"accuracy_per_step": np.array([]), "volume_per_step": np.array([]),
                "n_evaluated": 0}

    N = min(len(v) for v in reachable_sets.values())
    acc   = np.zeros(N)
    vol   = np.zeros(N)
    count = np.zeros(N, dtype=int)

    for veh_id, zonos in reachable_sets.items():
        if veh_id not in ground_truth:
            continue
        traj_gt = ground_truth[veh_id]

        for k in range(min(N, len(traj_gt))):
            zono = zonos[k]
            area = zonotope_area(zono)
            if area > max_area:
                continue

            inside = int(is_inside(zono, traj_gt[k]))
            acc[k]   += inside
            vol[k]   += area
            count[k] += 1

    safe_count = np.where(count > 0, count, 1)
    return {
        "accuracy_per_step": acc / safe_count * 100.0,
        "volume_per_step":   vol / safe_count,
        "n_evaluated":       int(count.max()) if count.size else 0,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_inclusion_accuracy(
    results: dict,
    dt: float = 0.1,
    save_path: str = None,
):
    """Plot accuracy (%) and mean reachable set area over the time horizon.

    Parameters
    ----------
    results  : output of compute_inclusion_accuracy(...)
    dt       : timestep in seconds (default 0.1 s → 10 Hz)
    save_path: if given, save the figure here
    """
    import matplotlib.ticker as tck

    acc = results["accuracy_per_step"]
    vol = results["volume_per_step"]
    t   = np.arange(1, len(acc) + 1) * dt

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t, acc, "-", color="#4AB19D", linewidth=2, label="Inclusion Accuracy")
    ax1.set_xlabel("Time horizon (s)", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", color="#4AB19D", fontweight="bold")
    ax1.set_ylim([0, 110])
    ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax1.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax1.grid(which="major")
    ax1.grid(which="minor", ls="--", linewidth=0.3)
    ax1.tick_params(axis="y", labelcolor="#4AB19D")

    ax2 = ax1.twinx()
    ax2.fill_between(t, vol, alpha=0.15, color="#344E5C", label="Mean Reachable Set Area")
    ax2.set_ylabel("Area (m²)", color="#344E5C", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#344E5C")

    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="lower left")

    plt.title("Vehicle Reachable Set Inclusion Accuracy", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_state_inclusion_acc(config: dict, save_path: str = None):
    """Load saved reachable sets + ground truth from disk and evaluate.

    Expected pickle files (filled by Phase 3 simulation):
      <output_dir>/VANET/reachable_sets.pkl  → dict of zonotope lists
      <output_dir>/VANET/ground_truth.pkl    → dict of (T, 2) arrays
    """
    ra_path = os.path.join(config["output_dir"], "VANET", "reachable_sets.pkl")
    gt_path = os.path.join(config["output_dir"], "VANET", "ground_truth.pkl")

    with open(ra_path, "rb") as f:
        reachable_sets = pickle.load(f)
    with open(gt_path, "rb") as f:
        ground_truth = pickle.load(f)

    results = compute_inclusion_accuracy(reachable_sets, ground_truth)
    print(f"[Inclusion Accuracy] n_evaluated={results['n_evaluated']}")
    print(f"  Peak accuracy  : {results['accuracy_per_step'].max():.1f} %")
    print(f"  Area at horizon: {results['volume_per_step'][-1]:.2f} m²")

    visualize_inclusion_accuracy(
        results,
        save_path=save_path or os.path.join(config["output_dir"], "accuracy.png"),
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.clustering.run import load_config

    parser = argparse.ArgumentParser(description="VANET reachable set inclusion accuracy.")
    parser.add_argument("--folder",     type=str, default="experiments")
    parser.add_argument("--model_file", type=str, default="VANETDataset_pretrained")
    parser.add_argument("--index",      type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(folder=args.folder, model_file=args.model_file, index=args.index)
    get_state_inclusion_acc(cfg)
