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
        logger.info("Normalizing data ...")
        import numpy as np
        if hasattr(my_data, "all_chunks"):
            # ----------------------------------------------------------------
            # Global Standardization fitted ONLY on training data to prevent
            # data leakage.  Per-sample normalization is intentionally NOT used
            # here because absolute position/velocity values carry VANET lane
            # semantics that must be preserved across samples.
            # ----------------------------------------------------------------
            train_data_stack = np.concatenate(
                [my_data.all_chunks[i] for i in train_indices], axis=0
            ).astype(np.float64)  # upcast to float64 for stable statistics

            if config["data_normalization"] in ["standardization", "per_sample_std"]:
                mean = np.mean(train_data_stack, axis=0)   # (feat_dim,)
                std  = np.std(train_data_stack,  axis=0)   # (feat_dim,)
                # Guard: constant features (std==0) would produce NaN.
                # Use 1.0 so those features map to (x - mean)/1 = 0.
                safe_std = np.where(std < 1e-8, 1.0, std)
                logger.info("Global train mean per feature: {}".format(
                    dict(zip(my_data.feature_names, mean.tolist()))))
                logger.info("Global train std  per feature: {}".format(
                    dict(zip(my_data.feature_names, std.tolist()))))
                for i in range(len(my_data.all_chunks)):
                    my_data.all_chunks[i] = (
                        (my_data.all_chunks[i].astype(np.float64) - mean) / safe_std
                    ).astype(np.float32)

            elif config["data_normalization"] in ["minmax", "per_sample_minmax"]:
                min_val = np.min(train_data_stack, axis=0)
                max_val = np.max(train_data_stack, axis=0)
                safe_range = np.where((max_val - min_val) < 1e-8, 1.0, max_val - min_val)
                logger.info("Global train min per feature: {}".format(
                    dict(zip(my_data.feature_names, min_val.tolist()))))
                logger.info("Global train max per feature: {}".format(
                    dict(zip(my_data.feature_names, max_val.tolist()))))
                for i in range(len(my_data.all_chunks)):
                    my_data.all_chunks[i] = (
                        (my_data.all_chunks[i].astype(np.float64) - min_val) / safe_range
                    ).astype(np.float32)
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
        pin_memory=True,
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
