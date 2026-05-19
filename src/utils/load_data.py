from src.datasets.data import data_factory, Normalizer
from src.datasets.datasplit import split_dataset, save_indices
from torch.utils.data import DataLoader
from functools import partial
from src.datasets.masked_datasets import ImputationDataset, collate_unsuperv
import os
import torch


def load_task_datasets(config):
    """For the task specified in the configuration returns the corresponding combination of
    Task-specific Dataset class and collate function."""

    task = config["task"]

    if task == "imputation":
        return (
            partial(
                ImputationDataset,
                mean_mask_length=config["mean_mask_length"],
                masking_ratio=config["masking_ratio"],
                mode=config["mask_mode"],
                distribution=config["mask_distribution"],
                exclude_feats=config["exclude_feats"],
            ),
            collate_unsuperv,
        )
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def load_data(config, logger, save_data=True):
    """Load data, and split train and test dataset. If eval_only then only val_dataset will be created."""
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]
    my_data = data_class(config, n_proc=config["n_proc"])
    my_data.load_data()
    # my_data.tensor_3d.shape[0], (my_data.all_df.groupby(by='track_id').size()/60).sum() # TODO CHECK TENSOR

    # Split dataset
    if config["val_ratio"] == 1:
        val_indices = my_data.all_IDs
        train_indices = []
        logger.info("{} samples may be used for evaluation".format(len(val_indices)))
    else:
        train_indices, val_indices = split_dataset(
            data_indices=my_data.all_IDs,
            validation_ratio=config["val_ratio"],
            random_seed=config["seed"],
        )

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))

    # ── Memory guard: sub-sample for eval/clustering if requested ───────────
    eval_subset = config.get("eval_subset")
    if eval_subset is not None and eval_subset < len(val_indices):
        import random
        rng = random.Random(config.get("seed", 1337))
        val_indices = rng.sample(list(val_indices), eval_subset)
        logger.info(
            f"eval_subset={eval_subset}: sub-sampled val_indices to {len(val_indices)} chunks "
            f"(RAM ~{eval_subset * 60 * 51 * 4 / 1e9:.2f} GB)"
        )

    save_indices(
        indices={"train": train_indices, "val": val_indices},
        folder=config["output_dir"],
    )
    # Skip trying to index a NoneType dataframe, our custom data class handles it
    if hasattr(my_data, "feature_df") and my_data.feature_df is not None:
        train_data = my_data.feature_df.loc[train_indices]
        val_data = my_data.feature_df.loc[val_indices]
    else:
        # Pass a mock structure just to satisfy the Dataset constructor signature since
        # __getitem__ looks up arrays via my_data.all_chunks dynamically anyway
        train_data = my_data
        val_data = my_data

    if config["val_ratio"] == 1 and save_data:
        # save original data for evaluation
        outputs_filepath = os.path.join(
            os.path.join(config["output_dir"], "original_data.pt")
        )
        torch.save({"val_data": val_data}, outputs_filepath)

    # Pre-process features
    if config["data_normalization"] != "none":
        logger.info("Normalizing data (streaming, memory-efficient) ...")
        import numpy as np
        if hasattr(my_data, "all_chunks"):
            stat_indices = train_indices if len(train_indices) > 0 else val_indices

            if config["data_normalization"] in ["minmax", "per_sample_minmax"]:
                # --- Streaming min/max: O(1) memory regardless of dataset size ---
                first = my_data.all_chunks[stat_indices[0]].astype(np.float32)
                running_min = first.min(axis=0)   # (F,)
                running_max = first.max(axis=0)   # (F,)
                for i in stat_indices[1:]:
                    chunk = my_data.all_chunks[i].astype(np.float32)
                    running_min = np.minimum(running_min, chunk.min(axis=0))
                    running_max = np.maximum(running_max, chunk.max(axis=0))
                scale = (running_max - running_min) + 1e-8
                for i in range(len(my_data.all_chunks)):
                    my_data.all_chunks[i] = (
                        my_data.all_chunks[i].astype(np.float32) - running_min
                    ) / scale

            elif config["data_normalization"] in ["standardization", "per_sample_std"]:
                # --- Welford online algorithm: O(1) memory, numerically stable ---
                n = 0
                mean = np.zeros(my_data.all_chunks[stat_indices[0]].shape[1], dtype=np.float64)
                M2   = np.zeros_like(mean)
                for i in stat_indices:
                    chunk = my_data.all_chunks[i].astype(np.float64)  # (T, F)
                    for row in chunk:
                        n   += 1
                        delta = row - mean
                        mean += delta / n
                        M2   += delta * (row - mean)
                std = np.sqrt(M2 / max(n - 1, 1)).astype(np.float32)
                mean = mean.astype(np.float32)
                for i in range(len(my_data.all_chunks)):
                    my_data.all_chunks[i] = (
                        my_data.all_chunks[i].astype(np.float32) - mean
                    ) / (std + 1e-8)
        else:
            normalizer = Normalizer(config["data_normalization"])
            if len(train_indices):
                train_data = normalizer.normalize(train_data)
            if len(val_indices):
                val_data = normalizer.normalize(val_data)

    # Initialize data generators
    task_dataset_class, collate_fn = load_task_datasets(config)

    # Dataloaders
    val_dataset = task_dataset_class(val_data, val_indices)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
        collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),
    )

    train_loader = None
    if config["val_ratio"] < 1:
        train_dataset = task_dataset_class(train_data, train_indices)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),
        )

    return train_loader, val_loader, my_data
