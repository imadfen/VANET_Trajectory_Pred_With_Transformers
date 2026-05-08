"""
Hyperparameter tuning entry-point using Ray Tune + HyperOpt.

Key design decisions
--------------------
* `data_normalization` is **always** forced to "standardization" in every trial.
  Leaving it as "none" causes raw VANET coordinates (X ~ 4 400 m, Y ~ 5 000 m)
  to be fed directly into the MSE loss, which blows up to ~10^16.
* Global standardization is computed from training chunks only (see load_data.py)
  so there is no data leakage.
* The search space in hyperparameter_config still lists data_normalization as a
  choice (for forward-compatibility), but the `run()` function below hard-overrides
  it so a bad draw from the sampler cannot break training.
"""

import pprint as pp
import os

from src.options import Options
from src.utils.config_setup import setup
from main import run as main
from src.utils.hyperparemer_tuning_config import hyperparameter_config

import ray
from ray import tune, air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session


ROOT = os.getcwd()


def run(hyperparameter_config: dict):
    """Single Ray trial: parse args and delegate to main.run()."""

    # ------------------------------------------------------------------ #
    #  Hard-set flags that MUST NOT be touched by the search algorithm.   #
    # ------------------------------------------------------------------ #
    hyperparameter_config["data_dir"]   = ROOT + "/resources/VANET_data/raw/"
    hyperparameter_config["data_class"] = "sind"
    hyperparameter_config["pattern"]    = "data_car_"
    hyperparameter_config["pos_encoding"] = "learnable"
    hyperparameter_config["name"]       = "VANETDataset_pretrained"
    hyperparameter_config["comment"]    = (
        "pretraining_through_imputation-hyperparameter_tuning"
    )
    hyperparameter_config["output_dir"] = ROOT + "/ray_results"

    # CRITICAL: always apply global standardization.
    # Without this the raw X/Y/Speed values (~4 400 m, ~30 m/s) feed directly
    # into the MSE loss and produce losses ~10^16.
    hyperparameter_config["data_normalization"] = "standardization"

    args_list = [f"--{k}={v}" for k, v in hyperparameter_config.items()]
    args_list.append("--hyperparameter_tuning")
    args_list.append("--harden")

    opts = Options().parse(args_list)

    pp.pprint(vars(opts))

    main(setup(opts), session)


if __name__ == "__main__":
    import argparse
    import torch
    import multiprocessing

    # ------------------------------------------------------------------ #
    #  Optional smoke-test mode: python hyperparamer_tuning.py --smoke    #
    #  Runs 1 trial × 1 epoch so you can verify loss < 100 in ~60 s.     #
    # ------------------------------------------------------------------ #
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 trial for 1 epoch as a sanity / smoke test.",
    )
    smoke_args, _ = cli.parse_known_args()

    N_ITER = 1 if smoke_args.smoke else 1000

    os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # Disable Ray's aggressive memory killer
    ray.init()

    # When smoking, patch epochs=1 into the config so the trial exits fast.
    if smoke_args.smoke:
        from hyperopt import hp as _hp
        smoke_space = dict(hyperparameter_config)  # shallow copy
        smoke_space["epochs"] = _hp.choice("epochs", [1])
        smoke_space["batch_size"] = _hp.choice("batch_size", [32])
        active_space = smoke_space
    else:
        active_space = hyperparameter_config

    searcher = HyperOptSearch(
        space=active_space,
        metric="loss",
        mode="min",
        n_initial_points=max(1, int(N_ITER / 10)),
    )

    num_gpus   = 1 if torch.cuda.is_available() else 0
    # usable_cpus = max(1, multiprocessing.cpu_count() - 1)
    usable_cpus = 1

    algo = ConcurrencyLimiter(searcher, max_concurrent=1)
    objective = tune.with_resources(
        tune.with_parameters(run),
        resources={"cpu": usable_cpus, "gpu": num_gpus},
    )

    tuner = tune.Tuner(
        trainable=objective,
        run_config=air.RunConfig(
            name="VANET_tune",
            storage_path=os.path.abspath("./ray_results/"),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=False,
                checkpoint_frequency=0,
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

    if smoke_args.smoke:
        best = results.get_best_result(metric="loss", mode="min")
        best_loss = best.metrics.get("loss", float("inf"))
        print(f"\n[SMOKE TEST] Best trial loss = {best_loss:.4f}")
        if best_loss < 100:
            print("[SMOKE TEST] PASSED ✓  loss < 100")
        else:
            print("[SMOKE TEST] FAILED ✗  loss >= 100 — check normalization and NaN handling")
