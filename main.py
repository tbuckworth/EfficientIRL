import time

import pandas as pd
from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail

import eirl
import gymnasium as gym
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env


def eirl_constructor(env, expert_transitions, expert_rollouts, rng, learner):
    return eirl.EIRL(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=rng,
        ), {"n_epochs": 2}
def bc_constructor(env, expert_transitions, expert_rollouts, rng, learner):
    return bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=rng,
        ), {"n_epochs": 2}

def gail_constructor(env, expert_transitions, expert_rollouts, rng, learner):
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    return gail.GAIL(
        demonstrations=expert_rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    ), {"total_timesteps": 20_000}

algos = {
    "GAIL": gail_constructor,
    "EIRL": eirl_constructor,
    "BC": bc_constructor,
}

def main():
    epochs = 20
    # env = gym.make("CartPole-v1")
    # expert = PPO(
    #     policy=MlpPolicy,
    #     env=env,
    #     seed=0,
    #     batch_size=64,
    #     ent_coef=0.0,
    #     learning_rate=0.0003,
    #     n_epochs=10,
    #     n_steps=64,
    # )
    # expert.learn(10_000)  # set to 100_000 for better performance

    SEED = 42

    env = make_vec_env(
        "seals:seals/CartPole-v0",
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals/CartPole-v0",
        venv=env,
    )
    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=5,
        seed=SEED,
    )
    rng = np.random.default_rng()
    # expert_rollouts = rollout.rollout(
    #     expert,
    #     DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    #     rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    #     rng=rng,
    # )
    expert_rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=np.random.default_rng(SEED),
    )
    expert_transitions = rollout.flatten_trajectories(expert_rollouts)
    rew_track = {}
    expert_rewards, _ = evaluate_policy(
        expert, env, 10, return_episode_rewards=True
    )
    print(f"Expert Rewards: {np.mean(expert_rewards)}")
    rew_track["Expert"] = {"rewards": np.mean(expert_rewards), "elapsed": None}
    outputs = []
    for algo in algos.keys():
        expert_trainer, unit_multiplier = algos[algo](env, expert_transitions, expert_rollouts, rng, learner)
        cum_time = 0
        for epoch in range(1, epochs+1):
            start = time.time()
            expert_trainer.train(**unit_multiplier)
            elapsed = time.time() - start
            cum_time += elapsed
            rewards, _ = evaluate_policy(
                expert_trainer.policy, env, 10, return_episode_rewards=True
            )
            outputs += [{
                "mean_reards": np.mean(rewards),
                "std_rewards": np.std(rewards),
                "elapsed": elapsed,
                "total_time": cum_time,
                "epoch": epoch,
                "algo": algo,
                "unit_multiplier": unit_multiplier,
              }]
            print(f"{algo} Rewards: {np.mean(rewards):.2f}\t elapsed:{elapsed:.2f}")
            # rew_track[algo] = {"rewards": np.mean(rewards), "elapsed": elapsed}

    df = pd.DataFrame(outputs)
    print(df)
    df.to_csv("data/EIRL_times.csv")

    # expert_eirl_trainer = eirl.EIRL(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     demonstrations=expert_transitions,
    #     rng=rng,
    # )
    #
    # expert_bc_trainer = bc.BC(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     demonstrations=expert_transitions,
    #     rng=rng,
    # )
    #
    # expert_bc_trainer.train(n_epochs=20)
    # expert_eirl_trainer.train(n_epochs=20)

    # expert_rewards, _ = evaluate_policy(
    #     expert.policy, env, 10, return_episode_rewards=True
    # )
    # print(f"Expert Rewards: {np.mean(expert_rewards)}")
    #
    # bc_expert_rewards, _ = evaluate_policy(
    #     expert_bc_trainer.policy, env, 10, return_episode_rewards=True
    # )
    # print(f"BC Rewards: {np.mean(bc_expert_rewards)}")
    #
    # eirl_expert_rewards, _ = evaluate_policy(
    #     expert_eirl_trainer.policy, env, 10, return_episode_rewards=True
    # )
    # print(f"EIRL Rewards: {np.mean(eirl_expert_rewards)}")


if __name__ == '__main__':
    main()
