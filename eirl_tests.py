import os
import re
import unittest

import numpy as np
import torch
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import eirl
from ant_v1_learner_config import load_ant_ppo_learner, load_ant_sac_learner, load_ppo_learner
from callbacks import RewardLoggerCallback
from helper_local import import_wandb, get_config, load_env, get_policy_for, load_expert_transitions
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


def get_latest_model(folder, keyword):
    search = lambda x: re.search(rf"model_{keyword}_(\d*).pth", x)
    if search(folder):
        return folder
    files = [os.path.join(folder, x) for x in os.listdir(folder)]
    last_checkpoint = max([int(search(x).group(1)) for x in files if search(x)])
    return [x for x in files if re.search(f"model_{keyword}_{last_checkpoint}.pth", x)][0]


class TestHopperLearner(unittest.TestCase):
    def setUp(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logdir = "logs/train/seals/Hopper-v1/2025-03-18__11-37-00__seed_0/"
        model_file = get_latest_model(logdir, "SUP")
        cfg = get_config(model_file)
        env_name = cfg["env_name"]
        n_envs = cfg["n_envs"]
        seed = cfg["seed"]
        net_arch = cfg["net_arch"]
        default_rng, env = load_env(env_name, n_envs, seed)
        policy = get_policy_for(env.observation_space, env.action_space, net_arch)
        policy.to(device)
        policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])
        self.learner = load_ppo_learner(env_name, env, None, policy)
        self.policy = policy
        self.cfg = cfg
        self.model_file = model_file

    def test_learner(self):
        self.learner.learn(1000_000)

    def test_trainer(self):
        default_rng, env, expert_transitions, target_rewards = load_expert_transitions(
            self.cfg["env_name"],
            self.cfg["n_envs"],
            self.cfg["n_eval_episodes"],
            10,#self.cfg["n_expert_demos"],
            self.cfg["seed"]
        )
        expert_trainer = eirl.EIRL(
            policy=self.policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=default_rng,
            consistency_coef=self.cfg["consistency_coef"],
            hard=self.cfg["hard"],
            gamma=self.cfg["gamma"],
            batch_size=self.cfg["batch_size"],
            l2_weight=self.cfg["l2_weight"],
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": self.cfg["lr"]},
            use_next_state_reward=self.cfg["use_next_state_reward"],
            maximize_reward=self.cfg["maximize_reward"],
            log_prob_adj_reward=self.cfg["log_prob_adj_reward"],
        )
        expert_trainer.reward_func.load_state_dict(
            torch.load(self.model_file, map_location=self.policy.device
                       )["reward_func"])
        if self.cfg["use_next_state_reward"]:
            expert_trainer.state_reward_func.load_state_dict(
                torch.load(self.model_file, map_location=self.policy.device
                           )["state_reward_func"])
        if self.cfg["log_prob_adj_reward"]:
            expert_trainer.lp_adj_reward.load_state_dict(
                torch.load(self.model_file, map_location=self.policy.device
                           )["lp_adj_reward"])
        expert_trainer.train(n_epochs=5, progress_bar=False)



if __name__ == '__main__':
    unittest.main()
