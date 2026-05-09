import pprint as pp

from src.options import Options
from src.utils.config_setup import setup
from main import run as main
from src.utils.hyperparemer_tuning_config import hyperparameter_config

import ray
from ray import tune, air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session

import os


ROOT = os.getcwd()


def run(hyperparameter_config: dict):
    # Pretty print the run args
    hyperparameter_config["data_dir"] = ROOT + "/resources/VANET_data/raw/"
    hyperparameter_config["data_class"] = "sind"
    hyperparameter_config["pattern"] = "data_car_"
    hyperparameter_config["pos_encoding"] = "learnable"
    hyperparameter_config["name"] = "VANETDataset_pretrained"
    hyperparameter_config["comment"] = (
        "pretraining_through_imputation-hyperparameter_tuning"
    )
    hyperparameter_config["output_dir"] = ROOT + "/ray_results"

    args_list = [f"--{k}={v}" for k, v in hyperparameter_config.items()]
    args_list.append("--hyperparameter_tuning")
    args_list.append("--harden")

    opts = Options().parse(args_list)

    pp.pprint(vars(opts))

    main(setup(opts), session)


if __name__ == "__main__":
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # Disable Ray's aggressive memory killer
    N_ITER = 1000
    ray.init()  # Let Ray auto-detect available resources (e.g. Colab's 2 CPUs)
    searcher = HyperOptSearch(
        space=hyperparameter_config,
        metric="loss",
        mode="min",
        n_initial_points=int(N_ITER / 10),
    )
    import torch
    import multiprocessing
    
    num_gpus = 1 if torch.cuda.is_available() else 0
    # Automatically detect CPUs and leave 1 free for the OS/Ray background tasks
    usable_cpus = max(1, multiprocessing.cpu_count() - 1)
    
    algo = ConcurrencyLimiter(searcher, max_concurrent=1)  # Keep 1 trial at a time to prevent RAM crashes
    objective = tune.with_resources(
        tune.with_parameters(run), resources={"cpu": usable_cpus, "gpu": num_gpus}  # Allocate all safe CPUs to this single trial
    )

    experiment_dir = os.path.abspath("./ray_results/VANET_tune")

    if tune.Tuner.can_restore(experiment_dir):
        print(f"Resuming existing Tune experiment from: {experiment_dir}")
        tuner = tune.Tuner.restore(
            experiment_dir, 
            trainable=objective, 
            resume_unfinished=True, # Resume from where it was paused
            resume_errored=True     # Resume any errored trials from their latest checkpoint
        )
    else:
        tuner = tune.Tuner(
            trainable=objective,
            run_config=air.RunConfig(
                name="VANET_tune",
                storage_path=os.path.abspath("./ray_results/"),
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=1
                ),
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                search_alg=algo,
                num_samples=N_ITER,
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
        )

    results = tuner.fit()
    ray.shutdown()
