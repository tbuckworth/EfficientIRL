import unittest

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import eirl
from ant_v1_learner_config import load_ant_ppo_learner, load_ant_sac_learner, load_learner
from callbacks import RewardLoggerCallback
from helper_local import import_wandb, get_config, load_env, get_policy_for, load_expert_transitions, create_logdir, \
    get_latest_model
from eirl import load_expert_trainer
from train_EIRL import wrap_env_with_reward
import gymnasium as gym
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

    def test_ppo(self):
        wandb.init(project="EfficientIRL", sync_tensorboard=True)
        expert_trainer = eirl.EIRL(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=None,
            rng=self.rng,
        )
        logdir = create_logdir(self.env_name, 0)
        wenv = wrap_env_with_reward(self.env, expert_trainer.reward_func)
        learner = load_ant_ppo_learner(wenv, logdir, expert_trainer.policy)
        # for i in range(20):
        learner.learn(10_000, callback=RewardLoggerCallback())

    def test_sac(self):
        wandb.init(project="EfficientIRL", sync_tensorboard=True)
        expert_trainer = eirl.EIRL(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=None,
            rng=self.rng,
        )
        logdir = create_logdir(self.env_name, 0)
        wenv = wrap_env_with_reward(self.env, expert_trainer.reward_func)
        learner = load_ant_sac_learner(wenv, logdir, expert_trainer.policy)
        # for i in range(20):
        learner.learn(10_000, callback=RewardLoggerCallback())


class TestHopperLearner(unittest.TestCase):
    def setUp(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # logdir = "logs/train/seals:seals/MountainCar-v0/2025-03-19__07-57-24__seed_0"
        logdir = "logs/train/seals:seals/Hopper-v1/2025-03-20__13-47-46__seed_100"
        logdir = "logs/train/seals:seals/Hopper-v1/2025-03-21__05-14-17__seed_100"
        logdir = "logs/train/seals:seals/Hopper-v1/2025-03-21__10-24-57__seed_0"
        model_file = get_latest_model(logdir, "RL")
        cfg = get_config(model_file)
        env_name = cfg["env_name"]
        n_envs = cfg["n_envs"]
        record_video = True
        seed = cfg["seed"]
        net_arch = cfg["net_arch"]

        default_rng, env = load_env(env_name, n_envs, seed, )#{"render_mode":"human"})
        policy = get_policy_for(env.observation_space, env.action_space, net_arch)
        policy.to(device)
        policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])
        self.learner = load_learner(env_name, env, None, policy)
        self.policy = policy
        self.cfg = cfg
        self.model_file = model_file
        self.env = env
        if record_video:
            vid_env = gym.make(env_name, render_mode="rgb_array")
            # Set up video recording parameters
            video_folder = "./videos/"
            video_length = 1000  # adjust to the desired length (in timesteps)
            record_video_trigger = lambda x: x == 0  # records only the first episode; modify as needed

            # Wrap your vectorized environment with VecVideoRecorder
            self.vid_env = RecordVideo(
                vid_env,
                video_folder,
                episode_trigger = lambda e: True,
                video_length=video_length,
                name_prefix="1.25"
            )

    def test_record_video(self):
        obs, _ = self.vid_env.reset()
        cum_reward = 0
        for _ in range(1000):
            # Sample a random action
            with torch.no_grad():
                action = self.policy._predict(torch.FloatTensor(obs).to(device=self.policy.device).unsqueeze(0))

            # Step the environment
            obs, reward, term, trunc, info = self.vid_env.step(action.squeeze().cpu().numpy())
            cum_reward += reward
            # Render the *first* (index 0) sub-environment
            # This calls the 'render()' of the underlying gym environment

            # Optionally reset if done
            if term or trunc:
                print(f"Total Reward: {cum_reward}")
                self.vid_env.reset()
                break
        self.vid_env.close()

    def test_render(self):
        obs = self.env.reset()
        for _ in range(1000):
            # Sample a random action
            with torch.no_grad():
                action = self.policy._predict(torch.FloatTensor(obs).to(device=self.policy.device))

            # Step the environment
            obs, reward, done, info = self.env.step(action.cpu().numpy())

            # Render the *first* (index 0) sub-environment
            # This calls the 'render()' of the underlying gym environment
            self.env.envs[0].render()

            # Optionally reset if done
            if done[0]:
                self.env.reset()


    def test_learner(self):
        self.learner.learn(1000_000)

    def test_trainer(self):
        default_rng, env, expert_transitions, target_rewards = load_expert_transitions(
            self.cfg["env_name"],
            self.cfg["n_envs"],
            self.cfg["n_eval_episodes"],
            10,#self.cfg["n_expert_demos"],
            self.cfg["seed"],
            self.cfg["expert_algo"],
        )
        expert_trainer = load_expert_trainer(self.policy, self.cfg, self.model_file, default_rng, env,
                                             expert_transitions)
        expert_trainer.train(n_epochs=5, progress_bar=False)


if __name__ == '__main__':
    unittest.main()
