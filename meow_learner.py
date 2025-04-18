import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from helper_local import create_envs_meow_imitation_compat, create_logdir
from meow.meow_continuous_action import HybridPolicy, MEOW, create_envs_meow, FlowPolicy
from private_login import wandb_login
import wandb

def evaluate(env, expert_trainer):
    rewards, _ = evaluate_policy(
        expert_trainer.policy, env, 10, return_episode_rewards=True
    )
    return np.mean(rewards), None, None

def train_class(policy_constructor_type):
    env_name = "seals:seals/Hopper-v1"
    # env_name = "Hopper-v4"
    n_envs = 16
    norm_reward = False
    seed = 42
    tags = ["meow hybrid test"]
    cfg = locals()
    wandb_login()
    project = "EfficientIRL"
    logdir = create_logdir(env_name, seed)
    wandb.init(project=project, config=cfg, sync_tensorboard=True,
               tags=cfg["tags"], resume="allow")

    policy_constructor = get_policy_constructor(policy_constructor_type)

    envs, test_envs = create_envs_meow_imitation_compat(env_name, n_envs, norm_reward, seed)
    # envs, test_envs = create_envs_meow(env_name, seed, n_envs)
    learner = MEOW(envs, test_envs, policy_constructor=policy_constructor, evaluate=evaluate, logdir=logdir)
    learner.learn(1500_000, wandb=wandb)
    wandb.finish()


def get_policy_constructor(policy_constructor_type):
    if policy_constructor_type == "hybrid":
        return HybridPolicy
    elif policy_constructor_type == "flow":
        return FlowPolicy
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    for policy_constructor_type in ["flow", "hybrid"]:
        train_class(policy_constructor_type)
