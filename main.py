from imitation.algorithms import bc

import eirl
import gymnasium as gym
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    env = gym.make("CartPole-v1")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(10_000)  # set to 100_000 for better performance

    rng = np.random.default_rng()
    expert_rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    expert_transitions = rollout.flatten_trajectories(expert_rollouts)

    expert_eirl_trainer = eirl.EIRL(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_transitions,
        rng=rng,
    )
    expert_eirl_trainer.train(n_epochs=10)

    expert_bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_transitions,
        rng=rng,
    )

    expert_bc_trainer.train(n_epochs=10)

    expert_rewards, _ = evaluate_policy(
        expert.policy, env, 10, return_episode_rewards=True
    )
    print(f"Expert Rewards: {np.mean(expert_rewards)}")

    bc_expert_rewards, _ = evaluate_policy(
        expert_bc_trainer.policy, env, 10, return_episode_rewards=True
    )
    print(f"BC Rewards: {np.mean(bc_expert_rewards)}")

    eirl_expert_rewards, _ = evaluate_policy(
        expert_eirl_trainer.policy, env, 10, return_episode_rewards=True
    )
    print(f"EIRL Rewards: {np.mean(eirl_expert_rewards)}")


if __name__ == '__main__':
    main()
