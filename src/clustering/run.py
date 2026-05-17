"""
Phase 2 — VANET Behavior Clustering Pipeline

Orchestrates: load model → extract embeddings → HDBSCAN cluster → save AnnoyModel index.

Usage (from project root):
    python -m src.clustering.run \
        --folder experiments \
        --model_file VANETDataset_pretrained_2026-XX-XX_XX-XX-XX_XXX \
        --save_embeddings
"""

import torch
import numpy as np
import argparse
import os
import sys
import json
import logging

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_dir)

from main import run as run_transformer
from src.utils.config_setup import create_dirs
from src.clustering.Clusters import HDBSCANCluster
from src.clustering.NearestNeighbor import AnnoyModel
from src.utils.load_data import load_task_datasets
from src.transformer_model.model import create_model, evaluate
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

ROOT = os.getcwd()

# 51 VANET features matching SINDData.feature_names
VANET_FEATURE_NAMES = [
    'X', 'Y', 'Speed', 'Acceleration', 'Heading', 'AngularVelocity',
    'LaneID', 'LaneDist',
    'Neigh1_Rx', 'Neigh1_Ry', 'Neigh1_RSpeed', 'Neigh1_RHeading',
    'Neigh2_Rx', 'Neigh2_Ry', 'Neigh2_RSpeed', 'Neigh2_RHeading',
    'Neigh3_Rx', 'Neigh3_Ry', 'Neigh3_RSpeed', 'Neigh3_RHeading',
    'Neigh4_Rx', 'Neigh4_Ry', 'Neigh4_RSpeed', 'Neigh4_RHeading',
    'Neigh5_Rx', 'Neigh5_Ry', 'Neigh5_RSpeed', 'Neigh5_RHeading',
    'Neigh6_Rx', 'Neigh6_Ry', 'Neigh6_RSpeed', 'Neigh6_RHeading',
    'Neigh7_Rx', 'Neigh7_Ry', 'Neigh7_RSpeed', 'Neigh7_RHeading',
    'Neigh8_Rx', 'Neigh8_Ry', 'Neigh8_RSpeed', 'Neigh8_RHeading',
    'Neigh9_Rx', 'Neigh9_Ry', 'Neigh9_RSpeed', 'Neigh9_RHeading',
    'Neigh10_Rx', 'Neigh10_Ry', 'Neigh10_RSpeed', 'Neigh10_RHeading',
    'AvgDistToSender', 'AvgMsgDelay', 'PacketLossRate',
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings_from_pt(output_dir: str):
    """Load the per_batch tensors saved by the val evaluator."""
    pt_file = torch.load(os.path.join(output_dir, "output_data.pt"))

    all_embeddings = np.concatenate(pt_file["embeddings"], axis=0)
    all_predictions = np.concatenate(pt_file["predictions"], axis=0)
    all_targets = np.concatenate(pt_file["targets"], axis=0)
    padding_masks = np.concatenate(pt_file["padding_masks"], axis=0)
    target_masks = np.concatenate(pt_file["target_masks"], axis=0)

    # intent_logits may not be saved in older runs
    intent_logits = None
    if "intent_logits" in pt_file:
        intent_logits = np.concatenate(pt_file["intent_logits"], axis=0)

    return all_targets, all_embeddings, all_predictions, padding_masks, target_masks, intent_logits


def load_config(
    folder: str = "experiments",
    model_file: str = "VANETDataset_pretrained",
    index: int = 2,
    index_data: int = 0,
    original_data: bool = False,
) -> dict:
    config_path = os.path.join(folder, model_file, "configuration.json")
    with open(config_path) as f:
        config = json.load(f)

    config["save_dir"] = ROOT + f"/{folder}/" + config["save_dir"].split("/", index)[-1]
    config["output_dir"] = (
        ROOT + f"/{folder}/" + config["output_dir"].split("/", index)[-1] + "/eval"
    )
    config["tensorboard_dir"] = (
        ROOT + f"/{folder}/" + config["tensorboard_dir"].split("/", index)[-1] + "/eval"
    )
    config["data_dir"] = ROOT + "/" + config["data_dir"].split("/", index_data)[-1]
    config["load_model"] = os.path.join(config["save_dir"], "model_best.pth")
    config["eval_only"] = True
    config["save_embeddings"] = True
    config["val_ratio"] = 1.0
    config["dropout"] = 0.0
    config["hyperparameter_tuning"] = False

    create_dirs([config["output_dir"]])
    return config


# ---------------------------------------------------------------------------
# Nearest-neighbor cluster lookup (used by Phase 3 / simulation)
# ---------------------------------------------------------------------------

def get_embedding(config: dict, data_chunks: np.ndarray, chunk_indices: list):
    """Extract embeddings from the pre-trained encoder for a set of chunks.

    Returns the embedding array of shape (N, embedding_dim).
    """
    from src.datasets.masked_datasets import ImputationDataset, collate_unsuperv
    from functools import partial

    class _TmpData:
        """Minimal wrapper so ImputationDataset can address all_chunks."""
        def __init__(self, chunks):
            self.all_chunks = chunks

    task_dataset_class, collate_fn = load_task_datasets(config)

    tmp_data = _TmpData(data_chunks)
    dataset = task_dataset_class(tmp_data, chunk_indices)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,
        collate_fn=lambda x: collate_fn(x, max_len=config["data_chunk_len"]),
    )

    model, _, _, val_evaluator, _ = create_model(config, None, loader, tmp_data, logger, device="cpu")

    _, embedding_data = evaluate(val_evaluator, config=config, save_embeddings=True, save_data=False)
    return embedding_data["embeddings"][0]  # (N, embedding_dim)


def get_cluster(config: dict, embedding: np.ndarray):
    """Return the nearest cluster ID and distance for a single embedding vector.

    Parameters
    ----------
    config : dict  – must contain 'output_dir' pointing to where AnnoyModel was saved
    embedding : np.ndarray  – shape (embedding_dim,)

    Returns
    -------
    cluster_id : int
    distance : float
    """
    nn_model = AnnoyModel(config=config)
    return nn_model.get(embedding)


# ---------------------------------------------------------------------------
# Main clustering pipeline
# ---------------------------------------------------------------------------

def run_clusters(
    config: dict = None,
    load_embeddings: bool = True,
    load_clusters: bool = False,
    min_cluster_size: int = 5,
    min_samples: int = 30,
    save_data: bool = True,
    show_clusters: bool = True,
):
    """Full Phase 2 pipeline.

    1. If not load_embeddings: re-runs the Transformer in eval mode to extract them.
    2. Loads embeddings + targets from output_data.pt.
    3. Runs (or loads) HDBSCAN.
    4. Builds and saves the AnnoyModel nearest-neighbour index.
    """
    if not load_embeddings:
        run_transformer(config)

    (
        all_targets,
        all_embeddings,
        all_predictions,
        padding_masks,
        target_masks,
        intent_logits,
    ) = load_embeddings_from_pt(config["output_dir"])

    logger.info(
        f"Loaded embeddings: {all_embeddings.shape}, targets: {all_targets.shape}"
    )

    cluster_instance = HDBSCANCluster(
        embeddings=all_embeddings,
        target=all_targets,
        padding_masks=padding_masks,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        config=config,
    )

    if not load_clusters:
        cluster_instance.run(
            original_data=False,
            remove_noise=True,
            save_data=save_data,
            show_clusters=show_clusters,
        )
        # Build and save the Annoy nearest-neighbour index over cluster centroids
        nn_model = AnnoyModel(config=config)
        nn_model.build()
    else:
        data = cluster_instance.load_clusters(original_data=False, remove_noise=True)
        return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VANET behavior clustering (Phase 2).")
    parser.add_argument("--folder", type=str, default="experiments",
                        help="Experiments folder containing trained models.")
    parser.add_argument("--model_file", type=str, default="VANETDataset_pretrained",
                        help="Experiment sub-folder name (model to cluster).")
    parser.add_argument("--index", type=int, default=2,
                        help="Path split index for save_dir.")
    parser.add_argument("--index_data", type=int, default=0,
                        help="Path split index for data_dir.")
    parser.add_argument("--load_embeddings", action="store_true",
                        help="Load pre-extracted embeddings instead of re-running the encoder.")
    parser.add_argument("--load_clusters", action="store_true",
                        help="Load pre-computed HDBSCAN clusters.")
    parser.add_argument("--min_cluster_size", type=int, default=5)
    parser.add_argument("--min_samples", type=int, default=30)

    args = parser.parse_args()

    config = load_config(
        folder=args.folder,
        model_file=args.model_file,
        index=args.index,
        index_data=args.index_data,
    )

    run_clusters(
        config=config,
        load_embeddings=args.load_embeddings,
        load_clusters=args.load_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    logger.info("Phase 2 clustering finished.")
