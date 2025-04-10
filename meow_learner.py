import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from helper_local import create_envs_meow_imitation_compat, create_logdir
from meow.meow_continuous_action import HybridPolicy, MEOW, create_envs_meow
from private_login import wandb_login
import wandb

def evaluate(env, expert_trainer):
    rewards, _ = evaluate_policy(
        expert_trainer.policy, env, 10, return_episode_rewards=True
    )
    return np.mean(rewards), None, None

def train_class():
    env_name = "seals:seals/Hopper-v1"
    # env_name = "Hopper-v4"
    n_envs = 8
    norm_reward = False
    seed = 42
    tags = ["meow hybrid test"]
    cfg = locals()
    wandb_login()
    project = "EfficientIRL"
    logdir = create_logdir(env_name, seed)
    wandb.init(project=project, config=cfg, sync_tensorboard=True,
               tags=cfg["tags"], resume="allow")

    envs, test_envs = create_envs_meow_imitation_compat(env_name, n_envs, norm_reward, seed)
    # envs, test_envs = create_envs_meow(env_name, seed, n_envs)
    learner = MEOW(envs, test_envs, policy_constructor=HybridPolicy, evaluate=evaluate, logdir=logdir)
    learner.learn(100_000, wandb=wandb)

if __name__ == "__main__":
    train_class()
