"""
Vehicle driving-behavior label utilities for Phase 2 cluster visualisation.

Replaces the pedestrian SinD labeling oracle with vehicle intent labels
mapped to the 4-class intent head: MaintainLane / Turn / Exit / Brake.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Vehicle intent labels — aligned with encoder intent_head output classes
# ---------------------------------------------------------------------------
VEHICLE_LABELS = {
    0: "MaintainLane",
    1: "Turn",
    2: "Exit",
    3: "Brake",
}
REVERSED_VEHICLE_LABELS = {v: k for k, v in VEHICLE_LABELS.items()}

# 23 VANET feature names (indices 0=X, 1=Y, 2=Speed ...)
VANET_FEATURE_NAMES = [
    'X', 'Y', 'Speed', 'Acceleration', 'Heading', 'AngularVelocity',
    'LaneID', 'LaneDist',
    'Neigh1_Rx', 'Neigh1_Ry', 'Neigh1_RSpeed', 'Neigh1_RHeading',
    'Neigh2_Rx', 'Neigh2_Ry', 'Neigh2_RSpeed', 'Neigh2_RHeading',
    'Neigh3_Rx', 'Neigh3_Ry', 'Neigh3_RSpeed', 'Neigh3_RHeading',
    'AvgDistToSender', 'AvgMsgDelay', 'PacketLossRate',
]

COLORS = [
    "#EF3D59", "#E17A47", "#EFC958", "#4AB19D",
    "#344E5C", "#A6206A", "#568EA6", "#A2D4AB", "#5A5050",
]


# ---------------------------------------------------------------------------
# Colour palette helper
# ---------------------------------------------------------------------------

def get_color_palette(num_data: int):
    cmap = cm.get_cmap("hsv")
    return [cmap(i / max(num_data, 1)) for i in range(num_data)]


# ---------------------------------------------------------------------------
# Label assignment from intent logits
# ---------------------------------------------------------------------------

def assign_intent_labels(intent_logits: np.ndarray) -> np.ndarray:
    """Convert (N, 4) intent logit array to (N,) hard label indices.

    Parameters
    ----------
    intent_logits : np.ndarray  shape (N, 4)

    Returns
    -------
    labels : np.ndarray  shape (N,)  values in {0,1,2,3}
    """
    return np.argmax(intent_logits, axis=1)


# ---------------------------------------------------------------------------
# Trajectory grouping by label
# ---------------------------------------------------------------------------

def label_trajectories(
    data_original: np.ndarray,   # (N, seq_len, 23)
    padding_masks: np.ndarray,   # (N, seq_len)
    clusters: np.ndarray,        # (N,)
    labels: np.ndarray,          # (N,)
):
    """Group trajectories by their intent label.

    Returns
    -------
    trajectories : dict  { label_id -> { i -> (seq_len, 23) array } }
    padding_per_label : dict
    clusters_per_label : dict
    """
    unique_labels = np.unique(labels)
    trajectories = {lb: {} for lb in unique_labels}
    padding_per_label = {lb: {} for lb in unique_labels}
    clusters_per_label = {lb: {} for lb in unique_labels}

    for lb in unique_labels:
        mask = labels == lb
        traj_group = data_original[mask]       # (M, seq_len, 23)
        pad_group = padding_masks[mask]
        clust_group = clusters[mask]
        for i in range(len(traj_group)):
            trajectories[lb][i] = traj_group[i]          # (seq_len, 23)
            padding_per_label[lb][i] = pad_group[i]
            clusters_per_label[lb][i] = clust_group[i]

    return trajectories, padding_per_label, clusters_per_label


# ---------------------------------------------------------------------------
# Trajectory plots (vehicle — no map overlay)
# ---------------------------------------------------------------------------

def plot_trajectories_per_label(
    trajectories: dict,
    padding_masked: dict,
    color_palette=COLORS,
):
    """Plot X/Y trajectories grouped by vehicle intent label."""
    for label_id, label_name in VEHICLE_LABELS.items():
        if label_id not in trajectories:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(f"Vehicle trajectories — {label_name}", fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        color = color_palette[label_id % len(color_palette)]
        for i, traj in trajectories[label_id].items():
            mask = padding_masked[label_id].get(i)
            data = traj[mask] if mask is not None else traj
            x, y = data[:, 0], data[:, 1]
            ax.plot(x, y, c=color, alpha=0.6, linewidth=0.8)
            ax.scatter(x[0], y[0], c="green", s=8)
            ax.scatter(x[-1], y[-1], c="red", s=8)

        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_trajectories_color_clusters(
    trajectories: dict,
    padding_masked: dict,
    clusters_per_label: dict,
    label_id: int = 0,
    color_palette=None,
):
    """Plot one label group, colouring each trajectory by its HDBSCAN cluster."""
    if color_palette is None:
        all_clusters = np.concatenate([
            np.array(list(c.values())) for c in clusters_per_label.values()
        ])
        color_palette = get_color_palette(int(all_clusters.max()) + 2)

    label_name = VEHICLE_LABELS.get(label_id, str(label_id))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Cluster colours — {label_name}", fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    used = set()

    for i, traj in trajectories.get(label_id, {}).items():
        mask = padding_masked[label_id].get(i)
        data = traj[mask] if mask is not None else traj
        x, y = data[:, 0], data[:, 1]
        c_id = int(clusters_per_label[label_id].get(i, 0))
        lbl = f"Cluster {c_id}"
        if lbl not in used:
            ax.plot(x, y, c=color_palette[c_id % len(color_palette)], alpha=0.85, label=lbl)
            used.add(lbl)
        else:
            ax.plot(x, y, c=color_palette[c_id % len(color_palette)], alpha=0.85)
        ax.scatter(x[0], y[0], c="green", s=5)
        ax.scatter(x[-1], y[-1], c="red", s=5)

    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Dimensionality reduction plots (reusable — geometry-agnostic)
# ---------------------------------------------------------------------------

def plot_dual_tsne_3d(data_cluster1, data_cluster2, figsize=(14, 6)):
    data1, clusters1 = data_cluster1
    data2, clusters2 = data_cluster2
    if data1.ndim > 2:
        data1 = data1.reshape(data1.shape[0], -1)
    if data2.ndim > 2:
        data2 = data2.reshape(data2.shape[0], -1)

    tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    d1t = tsne.fit_transform(data1)
    d2t = tsne.fit_transform(data2)

    COLOR_PALETTE = get_color_palette(max(len(set(clusters1)), len(set(clusters2))))
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection="3d")
    sc1 = ax1.scatter(d1t[:, 0], d1t[:, 1], d1t[:, 2], c=clusters1,
                      cmap=ListedColormap(COLOR_PALETTE), edgecolor="k")
    ax1.legend(*sc1.legend_elements(), loc="upper right", title="Clusters")
    ax1.set_title("t-SNE of Original Data")

    ax2 = fig.add_subplot(122, projection="3d")
    sc2 = ax2.scatter(d2t[:, 0], d2t[:, 1], d2t[:, 2], c=clusters2,
                      cmap=ListedColormap(COLOR_PALETTE), edgecolor="k")
    ax2.legend(*sc2.legend_elements(), loc="upper right", title="Clusters")
    ax2.set_title("t-SNE of Embeddings")

    plt.tight_layout()
    plt.show()


def plot_dual_pca_3d(data_cluster1, data_cluster2, n_components=3,
                     figsize=(14, 6), file: str = "pca_plot"):
    data1, clusters1 = data_cluster1
    data2, clusters2 = data_cluster2
    if data1.ndim > 2:
        data1 = data1.reshape(data1.shape[0], -1)
    if data2.ndim > 2:
        data2 = data2.reshape(data2.shape[0], -1)

    pca = PCA(n_components=n_components)
    d1t = pca.fit_transform(data1)
    d2t = pca.fit_transform(data2)

    COLOR_PALETTE = get_color_palette(max(len(set(clusters1)), len(set(clusters2))))
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(d1t[:, 0], d1t[:, 1], d1t[:, 2], c=clusters1,
                cmap=ListedColormap(COLOR_PALETTE), edgecolor="k")
    ax1.set_title("PCA of Original Data", fontweight="bold", fontsize=14)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(d2t[:, 0], d2t[:, 1], d2t[:, 2], c=clusters2,
                cmap=ListedColormap(COLOR_PALETTE), edgecolor="k")
    ax2.set_title("PCA of Embeddings", fontweight="bold", fontsize=14)

    plt.subplots_adjust(wspace=0)
    plt.savefig(f"{file}.png", dpi=300, bbox_inches="tight")
    plt.show()
