"""Config files for tuning experiments."""

import ray.tune as tune
import sacred
from torch import nn

from imitation.algorithms import dagger as dagger_alg
from imitation.scripts.parallel import parallel_ex

tuning_ex = sacred.Experiment("tuning", ingredients=[parallel_ex])

# First, define a base config that declares parallel_run_config
@tuning_ex.config
def base_cfg():
    parallel_run_config = dict()  # just an empty dict for now
    environment = None  # or some default environment
    # You can also define placeholders for other top-level keys here if needed

# Next, define a config for environment
@tuning_ex.config
def environment_cfg(parallel_run_config, environment):
    # If environment was overridden from CLI, use it
    if environment is not None:
        # Make sure parallel_run_config has the structure you want
        parallel_run_config.setdefault("base_config_updates", {})
        parallel_run_config["base_config_updates"].setdefault("environment", {})
        parallel_run_config["base_config_updates"]["environment"]["gym_id"] = environment



@tuning_ex.named_config
def rl():
    parallel_run_config = dict(
        sacred_ex_name="train_rl",
        run_name="rl_tuning",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={"environment": {"num_vec": 1}},
        search_space={
            "config_updates": {
                "rl": {
                    "batch_size": tune.choice([512, 1024, 2048, 4096, 8192]),
                    "rl_kwargs": {
                        "learning_rate": tune.loguniform(1e-5, 1e-2),
                        "batch_size": tune.choice([64, 128, 256, 512]),
                        "n_epochs": tune.choice([5, 10, 20]),
                    },
                },
            },
        },
        num_samples=100,
        repeat=1,
        resources_per_trial=dict(cpu=1),
    )
    num_eval_seeds = 5


@tuning_ex.named_config
def eirl():
    parallel_run_config = dict(
        sacred_ex_name="train_eirl",
        run_name="eirl_tuning",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={
            "environment": {"num_vec": 1},
            "demonstrations": {"source": "huggingface"},
        },
        search_space={
            "config_updates": {
                "eirl": dict(
                    batch_size=tune.choice([8, 16, 32, 64]),
                    l2_weight=tune.loguniform(1e-6, 1e-2),  # L2 regularization weight
                    optimizer_kwargs=dict(
                        lr=tune.loguniform(1e-5, 1e-2),
                    ),
                    train_kwargs=dict(
                        n_epochs=tune.choice([1, 5, 10, 20]),
                    ),
                ),
            },
            "command_name": "eirl",
        },
        num_samples=64,
        repeat=3,
        resources_per_trial=dict(cpu=1),
    )

    num_eval_seeds = 5
    eval_best_trial_resource_multiplier = 1

