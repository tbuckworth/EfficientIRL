import argparse
import os.path
import time

import pandas as pd
from imitation.algorithms import bc, sqil, dagger
from imitation.algorithms.adversarial import gail, airl
from stable_baselines3.sac import sac

import eirl
import gymnasium as gym
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from plot_times import plot
SEED = 42

# Name ID Base
# Ant seals/Ant-v0 Ant-v3
# Half Cheetah seals/HalfCheetah-v0 HalfCheetah-v3
# Hopper seals/Hopper-v0 Hopper-v3
# Swimmer seals/Swimmer-v0 Swimmer-v3
# Walker seals/Walker2d-v0 Walker2d-v3

def read_ant():
    with open("ant.out", "r") as file:
        lines = file.readlines()
        [l for l in lines if l[:6]=="Epoch:"]

def eirl_constructor(env, expert_transitions, expert_rollouts, rng):
    return eirl.EIRL(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_transitions,
        rng=rng,
    ), {"n_epochs": 1}


def bc_constructor(env, expert_transitions, expert_rollouts, rng):
    return bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_transitions,
        rng=rng,
    ), {"n_epochs": 1}


def gail_constructor(env, expert_transitions, expert_rollouts, rng):
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
        allow_variable_horizon=True,
    ), {"total_timesteps": 40_000}

def airl_constructor(env, expert_transitions, expert_rollouts, rng):
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
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    return airl.AIRL(
        demonstrations=expert_rollouts,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    ), {"total_timesteps": 40_000}

def sqil_constructor(env, expert_transitions, expert_rollouts, rng):
    return sqil.SQIL(
        venv=env,
        demonstrations=expert_transitions,
        policy="MlpPolicy",
    ), {"total_timesteps": 100_000}

def dagger_constructor(env, expert_transitions, expert_rollouts, rng):
    raise NotImplementedError
    # return dagger.SimpleDAggerTrainer(
    #     venv=env,
    #     scratch_dir=tmpdir,
    #     expert_policy=expert,
    #     bc_trainer=bc_trainer,
    #     rng=np.random.default_rng(),
    # )

    # dagger_trainer.train(2000)

algos = {
    "SQIL": sqil_constructor,
    "AIRL": airl_constructor,
    "GAIL": gail_constructor,
    "EIRL": eirl_constructor,
    "BC": bc_constructor,
    # "DAgger": dagger_constructor,
}


def main(algo_list, filename="EIRL_times2", load_expert=True):
    if algo_list == ["ALL"]:
        algo_list = algos.keys()
    csv_file = f"data/{filename}.csv"
    output_file = f"data/{filename}.png"
    epochs = 50
    # env = gym.make("CartPole-v1")

    env = make_vec_env(
        "seals:seals/CartPole-v0",
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )
    threshold = 500
    if load_expert:
        expert = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name="seals/CartPole-v0",
            venv=env,
        )
    else:
        agent = PPO(
            policy=MlpPolicy,
            env=env,
            seed=0,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
        )
        expert_rewards = [0]
        while np.mean(expert_rewards)<100:
            agent.learn(10_000)  # set to 100_000 for better performance
            expert = agent.policy
            expert_rewards, _ = evaluate_policy(
                expert, env, 10, return_episode_rewards=True
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
    outputs = [{
        "mean_rewards": np.mean(expert_rewards),
        "std_rewards": np.std(expert_rewards),
        "std_error": np.std(expert_rewards) / np.sqrt(len(expert_rewards)),
        "elapsed": None,
        "total_time": 0,
        "epoch": None,
        "algo": "Expert",
        "unit_multiplier": None,
    }]
    print(f"Expert Rewards: {np.mean(expert_rewards)}")
    rew_track["Expert"] = {"rewards": np.mean(expert_rewards), "elapsed": None}
    for algo in algo_list:
        expert_trainer, unit_multiplier = algos[algo](env, expert_transitions, expert_rollouts, rng)
        cum_time = 0
        for epoch in range(1, epochs + 1):
            start = time.time()
            expert_trainer.train(**unit_multiplier)
            elapsed = time.time() - start
            cum_time += elapsed
            rewards, _ = evaluate_policy(
                expert_trainer.policy, env, 10, return_episode_rewards=True
            )
            outputs += [{
                "mean_rewards": np.mean(rewards),
                "std_rewards": np.std(rewards),
                "std_error": np.std(rewards) / np.sqrt(len(rewards)),
                "elapsed": elapsed,
                "total_time": cum_time,
                "epoch": epoch,
                "algo": algo,
                "unit_multiplier": unit_multiplier,
            }]
            print(f"{algo} Rewards: {np.mean(rewards):.2f}\t elapsed:{elapsed:.2f}")
            # rew_track[algo] = {"rewards": np.mean(rewards), "elapsed": elapsed}
            if epoch > 5 and np.mean(rewards)>=threshold:
                break
    try:
        outputs += [{
            "mean_rewards": np.mean(expert_rewards),
            "std_rewards": np.std(expert_rewards),
            "std_error": np.std(expert_rewards) / np.sqrt(len(expert_rewards)),
            "elapsed": None,
            "total_time": max([l["total_time"] for l in outputs]),
            "epoch": None,
            "algo": "Expert",
            "unit_multiplier": None,
        }]
    except Exception as e:
        pass
    df = pd.DataFrame(outputs)
    print(df)
    if not os.path.exists(csv_file):
        df.to_csv(csv_file)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)
    plot(csv_file, output_file)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_list', type=str, nargs='+', default=['EIRL'])
    parser.add_argument('--filename', type=str, default='EIRL_times_default')
    parser.add_argument('--load_expert', action="store_true", default=False)

    main(**vars(parser.parse_args()))
