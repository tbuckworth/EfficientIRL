import os

import numpy as np
from imitation.algorithms.adversarial import airl
from imitation.policies.base import FeedForward32Policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imit_logger
from stable_baselines3 import PPO

from helper_local import load_expert_transitions, load_expert_rollouts, create_logdir, import_wandb
import json

from train_EIRL import evaluate

wandb = import_wandb()


def train_AIRL(
        env_name,
        norm_reward=False,
        expert_algo="ppo",
        seed=42,
        n_expert_demos=60,
        n_envs=16,
        tags=None,
):
    if tags is None:
        tags = []
    with open('hp_tune/airl_seals_ant_best_hp_eval.json', 'r') as f:
        cfg = json.load(f)

    n_eval_episodes = cfg['policy_evaluation']['n_episodes_eval']
    n_disc_updates_per_round = cfg['algorithm_kwargs']['n_disc_updates_per_round']
    gen_replay_buffer_capacity = cfg['algorithm_kwargs']['gen_replay_buffer_capacity']
    demo_batch_size = cfg['algorithm_kwargs']['demo_batch_size']
    rl_kwargs = cfg['rl']['rl_kwargs']
    total_timesteps = cfg['total_timesteps']

    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), locals())
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=locals(), tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])

    default_rng, env, expert_rollouts, target_rewards = load_expert_rollouts(env_name, expert_algo, n_envs,
                                                                             n_eval_episodes, n_expert_demos,
                                                                             norm_reward, seed)
    policy = FeedForward32Policy()
    learner = PPO(
        env=env,
        policy=policy,
        seed=seed,
        **rl_kwargs,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    expert_trainer = airl.AIRL(
        demonstrations=expert_rollouts,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        custom_logger=custom_logger,
        allow_variable_horizon=False,  # TODO - what do we really want here?
    )
    expert_trainer.train(total_timesteps=total_timesteps)

    mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement", log=True)


if __name__ == '__main__':
    train_AIRL(env_name="seals:seals/Ant-v1")
