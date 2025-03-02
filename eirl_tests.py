import unittest

import numpy as np
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import eirl

SEED = 42


class MyTestCase(unittest.TestCase):
    def test_something(self):
        rng = np.random.default_rng(SEED)
        env_name = "seals/CartPole-v0"
        env_name = "Pendulum-v1"
        env = make_vec_env(
            f"seals:{env_name}",
            rng=rng,
            n_envs=8,
            post_wrappers=[
                lambda env, _: RolloutInfoWrapper(env)
            ],  # needed for computing rollouts later
        )
        expert = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name=env_name,
            venv=env,
        )
        expert_rewards, _ = evaluate_policy(
            expert, env, 10, return_episode_rewards=True
        )
        expert_rollouts = rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=60),
            rng=rng,
        )
        expert_transitions = rollout.flatten_trajectories(expert_rollouts)
        expert_trainer = eirl.EIRL(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=rng,
        )
        expert_trainer.train(n_epochs=20,progress_bar=False)
        rewards, _ = evaluate_policy(
            expert_trainer.policy, env, 10, return_episode_rewards=True
        )
        print(f"Rewards:{np.mean(rewards)}\tStdError:{np.std(rewards)/np.sqrt(len(rewards))}")
        print(f"Expert Rewards{np.mean(expert_rewards)}")


if __name__ == '__main__':
    unittest.main()
