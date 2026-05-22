from src.datasets.data import data_factory, Normalizer
from src.datasets.datasplit import split_dataset, save_indices
from torch.utils.data import DataLoader
from functools import partial
from src.datasets.masked_datasets import ImputationDataset, collate_unsuperv
import os
import torch


def load_task_datasets(config):
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
    """Load data, and split train and test dataset."""
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]
    my_data = data_class(config, n_proc=config["n_proc"])
    my_data.load_data()

    # ── Try to load saved indices from model folder first ──────────────────
    indices_path = None
    for candidate in [
        config.get("output_dir") or "",
        os.path.dirname(config.get("load_model") or ""),
        os.path.dirname(os.path.dirname(config.get("load_model") or "")),
    ]:
        if not candidate:
            continue
        for filename in ["data_indices.pt", "data_indices.json"]:
            candidate_path = os.path.join(candidate, filename)
            if os.path.exists(candidate_path):
                indices_path = candidate_path
                break
        if indices_path:
            break

    if indices_path is not None:
        logger.info(f"Loading saved data split from {indices_path}")
        if indices_path.endswith(".json"):
            import json
            with open(indices_path) as f:
                saved = json.load(f)
            train_indices = saved["train_indices"]
            val_indices   = saved["val_indices"]
            test_indices  = saved["test_indices"]
        else:
            saved         = torch.load(indices_path)
            train_indices = saved["train"]
            val_indices   = saved["val"]
            test_indices  = saved["test"]
        logger.info(
            f"Restored split: {len(train_indices)} train / "
            f"{len(val_indices)} val / {len(test_indices)} test"
        )

    # ── Otherwise fall back to fresh split ─────────────────────────────────
    elif config["val_ratio"] == 1:
        val_indices   = my_data.all_IDs
        train_indices = []
        test_indices  = []
    else:
        train_indices, val_indices, test_indices = split_dataset(
            data_indices=my_data.all_IDs,
            validation_ratio=config["val_ratio"],
            test_ratio=config.get("test_ratio", 0.1),
            random_seed=config["seed"],
        )

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    # ── Memory guard: sub-sample val for eval/clustering if requested ───────
    eval_subset = config.get("eval_subset")
    if eval_subset is not None:
        if eval_subset < len(val_indices):
            import random
            rng = random.Random(config.get("seed", 1337))
            val_indices = rng.sample(list(val_indices), eval_subset)
            logger.info(
                f"eval_subset={eval_subset}: subsampled val to {len(val_indices)} samples "
                f"(RAM ~{eval_subset * 60 * 51 * 4 / 1e9:.2f} GB)"
            )
        else:
            logger.info(
                f"eval_subset={eval_subset} >= val size ({len(val_indices)}), using full val set"
            )

    # Only save indices on fresh training — not eval, not clustering
    if indices_path is None and not config.get("eval_only") and eval_subset is None:
        save_indices(
            indices={"train": train_indices, "val": val_indices, "test": test_indices},
            folder=config["output_dir"],
        )

    # Skip trying to index a NoneType dataframe
    if hasattr(my_data, "feature_df") and my_data.feature_df is not None:
        train_data = my_data.feature_df.loc[train_indices]
        val_data   = my_data.feature_df.loc[val_indices]
        test_data  = my_data.feature_df.loc[test_indices] if len(test_indices) else []
    else:
        train_data = my_data
        val_data   = my_data
        test_data  = my_data

    if config["val_ratio"] == 1 and save_data:
        outputs_filepath = os.path.join(config["output_dir"], "original_data.pt")
        torch.save({"val_data": val_data}, outputs_filepath)

    # Pre-process features
    if config["data_normalization"] != "none":
        logger.info("Normalizing data ...")
        import numpy as np
        import gc

        if hasattr(my_data, "all_chunks"):

            # ── Try to load saved norm constants (eval/subset/clustering mode) ──
            norm_loaded = False
            norm_path = None
            for candidate in [
                config.get("output_dir") or "",
                os.path.dirname(config.get("load_model") or ""),
                os.path.dirname(os.path.dirname(config.get("load_model") or "")),
                os.path.dirname(indices_path) if indices_path else "",
            ]:
                if not candidate:
                    continue
                p = os.path.join(candidate, "norm_constants.npy")
                if os.path.exists(p):
                    norm_path = p
                    break

            if norm_path is not None:
                logger.info(f"Loading normalization constants from {norm_path}")
                norm = np.load(norm_path, allow_pickle=True).item()
                if "min_val" in norm:
                    my_data.min_val   = norm["min_val"]
                    my_data.max_val   = norm["max_val"]
                    my_data.norm_type = norm["norm_type"]
                    for i in range(len(my_data.all_chunks)):
                        my_data.all_chunks[i] = (
                            my_data.all_chunks[i].astype(np.float32) - my_data.min_val
                        ) / ((my_data.max_val - my_data.min_val) + 1e-8)
                elif "mean" in norm:
                    my_data.mean      = norm["mean"]
                    my_data.std       = norm["std"]
                    my_data.norm_type = norm["norm_type"]
                    for i in range(len(my_data.all_chunks)):
                        my_data.all_chunks[i] = (
                            my_data.all_chunks[i] - my_data.mean
                        ) / (my_data.std + 1e-8)
                logger.info(f"Restored normalization constants ({norm['norm_type']})")
                norm_loaded = True

            # ── Compute fresh if not loaded ─────────────────────────────────
            if not norm_loaded:
                norm_indices = train_indices if len(train_indices) > 0 else val_indices
                logger.info(
                    f"Computing normalization constants from "
                    f"{'train' if len(train_indices) > 0 else 'val (no train indices)'} "
                    f"set ({len(norm_indices)} samples)"
                )
                train_data_stack = np.concatenate(
                    [my_data.all_chunks[i] for i in norm_indices], axis=0
                )

                if config["data_normalization"] in ["standardization", "per_sample_std"]:
                    mean = np.mean(train_data_stack, axis=0)
                    std  = np.std(train_data_stack, axis=0)
                    del train_data_stack
                    gc.collect()
                    my_data.mean      = mean
                    my_data.std       = std
                    my_data.norm_type = config["data_normalization"]
                    for i in range(len(my_data.all_chunks)):
                        my_data.all_chunks[i] = (my_data.all_chunks[i] - mean) / (std + 1e-8)

                    # Save only on fresh training run
                    if indices_path is None and not config.get("eval_only") and eval_subset is None:
                        norm_save_path = os.path.join(config["output_dir"], "norm_constants.npy")
                        np.save(norm_save_path, {"mean": mean, "std": std,
                                                  "norm_type": config["data_normalization"]})
                        logger.info(f"Saved normalization constants to {norm_save_path}")

                elif config["data_normalization"] in ["minmax", "per_sample_minmax"]:
                    min_val = np.min(train_data_stack, axis=0)
                    max_val = np.max(train_data_stack, axis=0)
                    del train_data_stack
                    gc.collect()
                    my_data.min_val   = min_val
                    my_data.max_val   = max_val
                    my_data.norm_type = config["data_normalization"]
                    for i in range(len(my_data.all_chunks)):
                        my_data.all_chunks[i] = (
                            my_data.all_chunks[i].astype(np.float32) - min_val
                        ) / ((max_val - min_val) + 1e-8)

                    # Save only on fresh training run
                    if indices_path is None and not config.get("eval_only") and eval_subset is None:
                        norm_save_path = os.path.join(config["output_dir"], "norm_constants.npy")
                        np.save(norm_save_path, {"min_val": min_val, "max_val": max_val,
                                                  "norm_type": config["data_normalization"]})
                        logger.info(f"Saved normalization constants to {norm_save_path}")

        else:
            normalizer = Normalizer(config["data_normalization"])
            my_data.normalizer = normalizer
            if len(train_indices):
                train_data = normalizer.normalize(train_data)
            if len(val_indices):
                val_data = normalizer.normalize(val_data)
            if len(test_indices):
                test_data = normalizer.normalize(test_data)

    # Initialize data generators
    task_dataset_class, collate_fn = load_task_datasets(config)

    val_dataset = task_dataset_class(val_data, val_indices)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
        collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),
    )

    test_loader = None
    if len(test_indices) > 0:
        test_dataset = task_dataset_class(test_data, test_indices)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),
        )

    # Only build train_loader during actual training
    train_loader = None
    if len(train_indices) > 0 and not config.get("eval_only") and eval_subset is None:
        train_dataset = task_dataset_class(train_data, train_indices)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),
        )

    return train_loader, val_loader, test_loader, my_data