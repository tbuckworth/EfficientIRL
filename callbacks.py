import numpy as np
import re
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

import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import wandb


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0, reward_window=100):
        super().__init__(verbose)
        self.reward_window = reward_window  # number of episodes to average over
        # Buffers for computing running mean of episode returns
        self.original_ep_returns = deque(maxlen=reward_window)
        self.learned_ep_returns = deque(maxlen=reward_window)
        # Per-env accumulators for current episode
        self._orig_sum = None
        self._learned_sum = None
        self._step_rews = None
        self._rew_names = None

    def _on_training_start(self) -> None:
        """Initialize per-environment reward accumulators."""
        num_envs = self.training_env.num_envs  # number of parallel envs
        self._orig_sum = np.zeros(num_envs)
        self._learned_sum = np.zeros(num_envs)

    def custom_step(self, num_timesteps):
        self.num_timesteps = num_timesteps
        return self._on_step()

    def cust_training_start(self, locals_, globals_, n_envs) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        self.num_timesteps = 0
        self._orig_sum = np.zeros(n_envs)
        self._learned_sum = np.zeros(n_envs)

    def _on_step(self) -> bool:
        # infos, rewards, and dones for each parallel env at this step
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        each_rew = np.array([[v for k, v in info.items() if re.search("rew", k)] for info in infos]).T
        all_rew = np.concatenate((each_rew, rewards[None, ...]))
        if self._step_rews is None:
            self._step_rews = all_rew
            self._rew_names = [k for k, v in infos[0].items() if re.search("rew", k)]
        else:
            self._step_rews = np.concatenate((self._step_rews, all_rew),axis=-1)[-500:]
            if len(self._step_rews) > 495:
                corrs = np.corrcoef(self._step_rews)[:-1, -1].tolist()
                wandb.log({f"correls/{k}": v for k, v in zip(self._rew_names, corrs)})

        # Update reward sums for each env
        for i, info in enumerate(infos):
            # Add current step rewards to accumulators
            # Original reward is stored by RewardVecEnvWrapper in info:
            orig_rew = info.get("original_env_rew", 0.0)
            self._orig_sum[i] += float(orig_rew)
            # Learned reward is the reward seen by the agent (from env.step):
            self._learned_sum[i] += float(rewards[i])
            # Check if this env finished an episode at this step
            if dones[i]:
                # Episode done: record returns and reset accumulators
                ep_orig_return = self._orig_sum[i]
                ep_learned_return = self._learned_sum[i]
                self.original_ep_returns.append(ep_orig_return)
                self.learned_ep_returns.append(ep_learned_return)
                if self.verbose:
                    print(
                        f"[Episode End] Env {i}: Original Return={ep_orig_return:.2f}, Learned Return={ep_learned_return:.2f}")
                # Reset for next episode
                # info["episode"]["r"]
                self._orig_sum[i] = 0.0
                self._learned_sum[i] = 0.0
        # If any episodes finished, log the average returns over recent episodes
        episodes_logged = len(self.original_ep_returns)  # total episodes recorded so far
        if episodes_logged > 0 and ("episode" in infos[-1] or True in dones):
            # Compute mean over the window (or over all so far if fewer than window size)
            mean_orig = np.mean(self.original_ep_returns)
            mean_learned = np.mean(self.learned_ep_returns)
            std_err_orig = np.std(self.original_ep_returns) / np.sqrt(len(self.original_ep_returns))
            std_err_learned = np.std(self.learned_ep_returns) / np.sqrt(len(self.learned_ep_returns))
            corr = np.corrcoef(self.original_ep_returns, self.learned_ep_returns)[0, 1]
            # Log to Weights & Biases with current timestep as x-axis
            wandb.log({
                "reward/original_ep_return_mean": mean_orig,
                "reward/original_ep_return_std_err": std_err_orig,
                "reward/learned_ep_return_mean": mean_learned,
                "reward/learned_ep_return_std_err": std_err_learned,
                "reward/reward_function_corr": corr,
            }, step=self.num_timesteps)
        return True

# Example usage:
# env = make_my_env()  # your environment
# venv = DummyVecEnv([lambda: env])
# venv = RewardVecEnvWrapper(venv, reward_fn)  # wrap with learned reward
# model = PPO("MlpPolicy", venv, ...)
# wandb.init(project="my-project", config=...)
# model.learn(total_timesteps=100000, callback=RewardLoggerCallback())
