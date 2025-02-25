from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import imitation.data.types
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.adversarial import gail
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.fft import ifftshift
from torch.utils.data import BatchSampler, SubsetRandomSampler

from CustomActor import EfficientIRLPolicy
from misc_util import orthogonal_init
from model import MlpModelNoFinalRelu


# class EfficientIRLMLPPolicy(BasePolicy):
#     def __init__(self, observation_space, hidden_dims, action_space):
#         super(EfficientIRLMLPPolicy, self).__init__()
#         self.model = MlpModelNoFinalRelu(observation_space.shape[0], hidden_dims + [[action_space, 1]])
#         self.apply(orthogonal_init)
#
#     def forward(self, obs):
#         phi, psi = self.model(obs)
#         log_probs = F.log_softmax(phi, dim=1)
#         p = Categorical(logits=log_probs)
#         return p
#
#     def act(self, obs):
#         return self.forward(obs).sample()
#
#     def forward_raw(self, obs):
#         return self.model(obs)
#
#     def _predict(self, observation, deterministic: bool = False):
#         return self.get_distribution(observation).get_actions(deterministic=deterministic)

    # def predict(self, obs):
    #     with torch.no_grad():
    #         obs = torch.FloatTensor(obs).to(self.device)
    #         act = self.act(obs)
    #     return act.cpu().numpy()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class CustomIRL:
    def __init__(self, env, expert_demos):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env
        self.expert_demos = expert_demos
        self.policy = EfficientIRLPolicy(env.observation_space, env.action_space, linear_schedule(0.001))

    def train(self, num_epochs=100000):
        minibatch_size=8196
        batch_size = ...
        # vars(expert_transitions).keys()
        # dict_keys(['obs', 'acts', 'infos', 'next_obs', 'dones'])
        for epoch in range(num_epochs):
            generator = self.get_generator(minibatch_size, batch_size)
            for sample in generator:
                obs, acts, infos, next_obs, dones = sample




    def evaluate(self, num_episodes=10):
        """Evaluate the trained policy."""
        returns = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.policy.predict(obs)
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward

            returns.append(total_reward)

        print(f"Average Return: {np.mean(returns)}")
        return np.mean(returns)

    def get_generator(self, mini_batch_size, batch_size):
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)
        names = ["obs", "acts", "infos", "next_obs", "dones"]
        for indices in sampler:
            yield (self.process_data(name, indices) for name in names)

    def process_data(self, name, indices):
        return torch.FloatTensor(self.expert_demos[name][indices]).to(self.device)


if __name__ == "__main__":
    # Set up environment
    env = gym.make("CartPole-v1")

    # Generate expert demonstrations using PPO
    expert_policy = PPO("MlpPolicy", env, verbose=1)
    expert_policy.learn(total_timesteps=10000)

    # Generate expert rollouts
    expert_rollouts = rollout.rollout(
        expert_policy,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=5000, min_episodes=50),
        rng=np.random.default_rng(),
    )
    expert_transitions = rollout.flatten_trajectories(expert_rollouts)

    # Initialize and train the custom IRL algorithm
    custom_irl = CustomIRL(env, expert_transitions)
    custom_irl.train(num_steps=50000)

    # Evaluate the trained policy
    custom_irl.evaluate()
