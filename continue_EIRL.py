import os

import numpy as np
import torch

from ant_v1_learner_config import load_ppo_learner
from eirl_tests import get_latest_model
from helper_local import get_config, load_env, get_policy_for, load_expert_transitions, import_wandb
from train_EIRL import evaluate, create_logdir
from imitation.util import logger as imit_logger
wandb = import_wandb()

def main(model_dir, run_from, epochs, tags):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = get_latest_model(model_dir, run_from)
    cfg = get_config(model_file)
    env_name = cfg["env_name"]
    n_envs = cfg["n_envs"]
    seed = cfg["seed"]
    net_arch = cfg["net_arch"]

    n_eval_episodes = cfg["n_eval_episodes"]
    n_expert_demos = cfg["n_expert_demos"]
    expert_algo = cfg["expert_algo"]
    norm_reward = cfg.get("norm_reward", False)
    cfg.update(locals())
    # net_arch = [32, 32]
    default_rng, env, expert_transitions, target_rewards = load_expert_transitions(env_name, n_envs,
                                                                                   n_eval_episodes,
                                                                                   n_expert_demos, seed,
                                                                                   expert_algo,
                                                                                   norm_reward)
    policy = get_policy_for(env.observation_space, env.action_space, net_arch)
    policy.to(device)
    policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])

    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), cfg)
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=cfg, tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])

    for i in range(epochs):
        learner = load_ppo_learner(env_name, env, logdir, policy)
        mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement", log=True)
        learner.learn(total_timesteps=1000_000)




if __name__ == "__main__":
    model_dir = "logs/train/seals:seals/Hopper-v1/2025-03-20__13-47-46__seed_100"
    epochs = 10
    tags = ["Continue learning"]
    main(model_dir, run_from="RL", epochs=epochs, tags=tags)
