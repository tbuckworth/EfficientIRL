import os

import numpy as np
import torch

from ant_v1_learner_config import load_ppo_learner
from callbacks import RewardLoggerCallback
from eirl_tests import get_latest_model
from helper_local import get_config, load_env, get_policy_for, load_expert_transitions, import_wandb, \
    load_expert_trainer
from train_EIRL import evaluate, create_logdir, override_env_and_wrap_reward
from imitation.util import logger as imit_logger

wandb = import_wandb()


def main(model_dir, run_from, tags, learner_timesteps=5000_000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = get_latest_model(model_dir, run_from)
    sup_model_file = get_latest_model(model_dir, "SUP")
    cfg = get_config(model_file)
    env_name = cfg["env_name"]
    n_envs = cfg["n_envs"]
    seed = cfg["seed"]
    net_arch = cfg["net_arch"]

    n_eval_episodes = cfg["n_eval_episodes"]
    n_expert_demos = cfg["n_expert_demos"]
    expert_algo = cfg["expert_algo"]
    norm_reward = cfg.get("norm_reward", False)
    log_prob_adj_reward = cfg["log_prob_adj_reward"]
    neg_reward = cfg["neg_reward"]
    override_env_name = cfg["override_env_name"]
    overrides = cfg["overrides"]
    rew_const_adj = cfg["rew_const_adj"]

    cfg.update({
        "model_dir": model_dir,
        "run_from": run_from,
        "learner_timesteps": learner_timesteps,
    })
    # net_arch = [32, 32]
    default_rng, env, expert_transitions, target_rewards = load_expert_transitions(env_name, n_envs,
                                                                                   n_eval_episodes,
                                                                                   n_expert_demos, seed,
                                                                                   expert_algo,
                                                                                   norm_reward)
    policy = get_policy_for(env.observation_space, env.action_space, net_arch)
    policy.to(device)
    policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])
    expert_trainer = load_expert_trainer(policy, cfg, sup_model_file, default_rng, env, expert_transitions)
    env, wenv = override_env_and_wrap_reward(env, env_name, expert_trainer, log_prob_adj_reward, n_envs, neg_reward,
                                             override_env_name, overrides, rew_const_adj)

    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), cfg)
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=cfg, tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])

    if run_from == "RL":
        learner = load_ppo_learner(env_name, wenv, logdir, policy)
        mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement", log=True)

        learner.learn(total_timesteps=learner_timesteps, callback=RewardLoggerCallback())
        torch.save({'model_state_dict': learner.policy.state_dict()},
                   f'{logdir}/model_RL_{learner_timesteps}.pth')
    else:
        raise NotImplementedError

    mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement", log=True)
    env.close()
    wandb.finish()


if __name__ == "__main__":
    model_dir = "logs/train/seals:seals/Hopper-v1/2025-03-20__13-47-46__seed_100"
    tags = ["Continue learning"]
    main(model_dir, run_from="RL", tags=tags)
