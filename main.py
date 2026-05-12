#!/usr/bin/env python
import logging
import sys
import os
import time
import torch
import logging

from src.options import Options
from src.utils import setup, load_data, register_record, readable_time
from src.transformer_model.model import create_model, evaluate, train

ROOT = os.getcwd()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def run(config, session=None):
    total_start_time = time.time()

    # Set the random seed
    if config["seed"] is not None:
        torch.manual_seed(config["seed"])

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)
    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))  # command used to run

    # Set the device
    use_cuda = torch.cuda.is_available() and not config["no_cuda"]
    use_mps = torch.backends.mps.is_available() and not config["no_cuda"]
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    ## Build and split data
    train_loader, val_loader, test_loader, data = load_data(config, logger, save_data=True)

    # Create model
    model, optimizer, trainer, val_evaluator, test_evaluator, start_epoch = create_model(
        config, train_loader, val_loader, test_loader, data, logger, device
    )

    if config["eval_only"]:
        logger.info("Evaluating model ...")
        evaluate(val_evaluator, config, save_embeddings=config["save_embeddings"], save_data=True)
        if test_evaluator is not None:
            logger.info("Evaluating on Test Set ...")
            aggr_metrics_test, per_batch_test = evaluate(test_evaluator, config, save_embeddings=config["save_embeddings"], save_data=True)
            # Dump test predictions to a CSV for charting
            import pandas as pd
            import numpy as np
            
            logger.info("Exporting test predictions to csv ...")
            preds = np.concatenate(per_batch_test["predictions"], axis=0)
            targets = np.concatenate(per_batch_test["targets"], axis=0)
            masks = np.concatenate(per_batch_test["padding_masks"], axis=0)
            
            # Fetch IDs
            ids_flat = []
            for b_ids in per_batch_test["IDs"]:
                ids_flat.extend(b_ids)
                
            # Inverse Normalization logic
            if hasattr(data, 'min_val') and data.min_val is not None:
                preds = preds * (data.max_val - data.min_val + 1e-8) + data.min_val
                targets = targets * (data.max_val - data.min_val + 1e-8) + data.min_val
            elif hasattr(data, 'mean') and data.mean is not None:
                preds = preds * (data.std + 1e-8) + data.mean
                targets = targets * (data.std + 1e-8) + data.mean
                
            records = []
            # Assume X is index 0, Y is index 1
            for i in range(len(preds)):
                track_id = ids_flat[i]
                active_len = int(np.sum(masks[i]))
                true_traj = targets[i, :active_len, :2]
                pred_traj = preds[i, :active_len, :2]
                
                if active_len > 0:
                    distances = np.linalg.norm(true_traj - pred_traj, axis=1)
                    ade = np.mean(distances)
                    fde = distances[-1]
                    rmse = np.sqrt(np.mean(distances**2))
                else:
                    distances, ade, fde, rmse = [], 0, 0, 0
                
                for t in range(active_len):
                    records.append({
                        "Trajectory_ID": track_id,
                        "Time_Step": t,
                        "True_X": true_traj[t, 0],
                        "True_Y": true_traj[t, 1],
                        "Pred_X": pred_traj[t, 0],
                        "Pred_Y": pred_traj[t, 1],
                        "Error_Delta": distances[t],
                        "ADE_Track": ade,
                        "FDE_Track": fde,
                        "RMSE_Track": rmse,
                        "Bounds_Area": 0.0, # Target Zonotope Reachable Set integration goes here
                        "Missed_Boolean": False # Target Reachability evaluation Boolean goes here
                    })
            
            df_export = pd.DataFrame(records)
            export_path = os.path.join(config["output_dir"], f"{config['experiment_name']}_test_predictions_charting.csv")
            df_export.to_csv(export_path, index=False)
            logger.info(f"Charts data exported to {export_path}")
            
            # Print global spatial metric outcomes over Test
            if len(df_export) > 0:
                global_ade = df_export.drop_duplicates(subset=["Trajectory_ID"])["ADE_Track"].mean()
                global_fde = df_export.drop_duplicates(subset=["Trajectory_ID"])["FDE_Track"].mean()
                global_rmse = df_export.drop_duplicates(subset=["Trajectory_ID"])["RMSE_Track"].mean()
                logger.info(f"--- SPATIAL TEST METRICS: ADE: {global_ade:.4f}m | FDE: {global_fde:.4f}m | RMSE: {global_rmse:.4f}m ---")
    else:
        logger.info("Starting training...")

        # Train Model
        aggr_metrics_val, best_metrics, best_value = train(
            model,
            optimizer,
            start_epoch,
            trainer,
            val_evaluator,
            train_loader,
            val_loader,
            config,
            session,
        )

        if test_evaluator is not None:
            logger.info("Running Final Evaluation on Test Set ...")
            aggr_metrics_test, per_batch_test = evaluate(test_evaluator, config, save_embeddings=config["save_embeddings"], save_data=True)
            
            import pandas as pd
            import numpy as np
            
            logger.info("Exporting test predictions to csv ...")
            preds = np.concatenate(per_batch_test["predictions"], axis=0)
            targets = np.concatenate(per_batch_test["targets"], axis=0)
            masks = np.concatenate(per_batch_test["padding_masks"], axis=0)
            
            # Fetch IDs
            ids_flat = []
            for b_ids in per_batch_test["IDs"]:
                ids_flat.extend(b_ids)
                
            # Inverse Normalization logic
            if hasattr(data, 'min_val') and data.min_val is not None:
                preds = preds * (data.max_val - data.min_val + 1e-8) + data.min_val
                targets = targets * (data.max_val - data.min_val + 1e-8) + data.min_val
            elif hasattr(data, 'mean') and data.mean is not None:
                preds = preds * (data.std + 1e-8) + data.mean
                targets = targets * (data.std + 1e-8) + data.mean
                
            records = []
            # Assume X is index 0, Y is index 1
            for i in range(len(preds)):
                track_id = ids_flat[i]
                active_len = int(np.sum(masks[i]))
                true_traj = targets[i, :active_len, :2]
                pred_traj = preds[i, :active_len, :2]
                
                if active_len > 0:
                    distances = np.linalg.norm(true_traj - pred_traj, axis=1)
                    ade = np.mean(distances)
                    fde = distances[-1]
                    rmse = np.sqrt(np.mean(distances**2))
                else:
                    distances, ade, fde, rmse = [], 0, 0, 0
                
                for t in range(active_len):
                    records.append({
                        "Trajectory_ID": track_id,
                        "Time_Step": t,
                        "True_X": true_traj[t, 0],
                        "True_Y": true_traj[t, 1],
                        "Pred_X": pred_traj[t, 0],
                        "Pred_Y": pred_traj[t, 1],
                        "Error_Delta": distances[t],
                        "ADE_Track": ade,
                        "FDE_Track": fde,
                        "RMSE_Track": rmse,
                        "Bounds_Area": 0.0, # Target Zonotope Reachable Set integration goes here
                        "Missed_Boolean": False # Target Reachability evaluation Boolean goes here
                    })
            
            df_export = pd.DataFrame(records)
            export_path = os.path.join(config["output_dir"], f"{config['experiment_name']}_test_predictions_charting.csv")
            df_export.to_csv(export_path, index=False)
            logger.info(f"Charts data exported to {export_path}")
            
            # Print global spatial metric outcomes over Test
            if len(df_export) > 0:
                global_ade = df_export.drop_duplicates(subset=["Trajectory_ID"])["ADE_Track"].mean()
                global_fde = df_export.drop_duplicates(subset=["Trajectory_ID"])["FDE_Track"].mean()
                global_rmse = df_export.drop_duplicates(subset=["Trajectory_ID"])["RMSE_Track"].mean()
                logger.info(f"--- SPATIAL TEST METRICS: ADE: {global_ade:.4f}m | FDE: {global_fde:.4f}m | RMSE: {global_rmse:.4f}m ---")

        # Export record metrics to a file accumulating records from all experiments in the same root file
        register_record(
            config["records_file"],
            config["initial_timestamp"],
            config["experiment_name"],
            best_metrics,
            aggr_metrics_val,
            comment=config["comment"],
        )

        logger.info(
            "Best loss was {}. Other metrics: {}".format(best_value, best_metrics)
        )
        logger.info("All Done!")
        logger.info(
            "Total runtime: {} hours, {} minutes, {} seconds\n".format(
                *readable_time(time.time() - total_start_time)
            )
        )

        return best_value


if __name__ == "__main__":
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    run(config)
