import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomVecEpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_envs = None  # To be initialized later
        self.episode_rewards = None
        self.total_rewards = []

    def _on_training_start(self):
        # Get number of environments
        self.n_envs = self.training_env.num_envs
        self.episode_rewards = np.zeros(self.n_envs)  # Track rewards per env

    def _on_step(self) -> bool:
        # Get rewards and done flags for all environments
        rewards = self.locals["infos"]["original_env_rew"]  # shape: (n_envs,)
        dones = self.locals["dones"]  # shape: (n_envs,)

        # Update per-environment statistics
        self.episode_rewards += rewards

        # Handle resets when an episode finishes in any env
        for i in range(self.n_envs):
            if dones[i]:  # Episode finished in env i
                self.total_rewards.append(self.episode_rewards[i])

                if self.verbose > 0:
                    print(f"Env {i} finished: Reward={self.episode_rewards[i]}")

                # Reset episode statistics for that environment
                self.episode_rewards[i] = 0

        return True

    def _on_training_end(self):
        # Print final statistics
        avg_reward = np.mean(self.total_rewards)
        print(f"Training Finished - Avg Reward: {avg_reward}")

#
# # Usage in SB3 training
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# import gym
#
# # Create environment and wrap it with VecMonitor for proper logging
# env = VecMonitor(DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)]))
#
# model = PPO("MlpPolicy", env, verbose=1)
#
# callback = CustomVecEpisodeStatsCallback(verbose=1)
# model.learn(total_timesteps=10000, callback=callback)
