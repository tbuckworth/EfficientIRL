import unittest

import numpy as np
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import eirl
from ant_v1_learner_config import load_ant_learner
from callbacks import RewardLoggerCallback
from helper_local import import_wandb
from train_EIRL import WandbInfoLogger, wrap_env_with_reward, create_logdir
wandb = import_wandb()

SEED = 42


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        # env_name = "seals/CartPole-v0"
        self.env_name = "seals/Ant-v1"
        self.env = make_vec_env(
            f"seals:{self.env_name}",
            rng=self.rng,
            n_envs=8,
            post_wrappers=[
                lambda env, _: RolloutInfoWrapper(env)
            ],  # needed for computing rollouts later
        )
        self.expert = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name=self.env_name,
            venv=self.env,
        )
    def test_something(self):

        expert_rewards, _ = evaluate_policy(
            self.expert, self.env, 10, return_episode_rewards=True
        )
        expert_rollouts = rollout.rollout(
            self.expert,
            self.env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=60),
            rng=self.rng,
        )
        expert_transitions = rollout.flatten_trajectories(expert_rollouts)
        expert_trainer = eirl.EIRL(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=expert_transitions,
            rng=self.rng,
        )
        expert_trainer.train(n_epochs=20,progress_bar=False)
        rewards, _ = evaluate_policy(
            expert_trainer.policy, self.env, 10, return_episode_rewards=True
        )
        print(f"Rewards:{np.mean(rewards)}\tStdError:{np.std(rewards)/np.sqrt(len(rewards))}")
        print(f"Expert Rewards{np.mean(expert_rewards)}")

    def test_other(self):
        wandb.init(project="EfficientIRL", sync_tensorboard=True)
        expert_trainer = eirl.EIRL(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=None,
            rng=self.rng,
        )
        logdir = create_logdir(self.env_name, 0)
        wenv = wrap_env_with_reward(self.env, expert_trainer.policy)
        learner = load_ant_learner(wenv, logdir)
        # for i in range(20):
        learner.learn(10_000, callback=RewardLoggerCallback())

if __name__ == '__main__':
    unittest.main()
