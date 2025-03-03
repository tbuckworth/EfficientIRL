import os

import numpy as np
from imitation.util.util import make_seeds
from stable_baselines3.common.monitor import Monitor

import time
from typing import Any, Dict, SupportsFloat, Tuple, List, Optional, Sequence, Callable, Mapping
from gymnasium.core import ActType, ObsType
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import functools

class CustomEnvMonitor(Monitor):
    original_rewards: List[float] = []
    episode_orig_returns: List[float] = []
    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        original_env_rew = float(info.get("original_env_rew", np.nan))
        self.original_rewards.append(original_env_rew)
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            orig_ep_rew = sum(self.original_rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6),
                       "l": ep_len,
                       "t": round(time.time() - self.t_start, 6),
                       "original_r": orig_ep_rew}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_orig_returns.append(orig_ep_rew)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.original_rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset()

    def get_episode_orig_rewards(self) -> List[float]:
        """
        Returns the original rewards of all the episodes

        :return:
        """
        return self.episode_orig_returns



class ExternalEnvMonitor(Monitor):
    original_rewards: List[float] = []
    episode_orig_returns: List[float] = []
    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        original_env_rew = float(info["original_env_rew"])
        self.original_rewards.append(original_env_rew)
        if done:
            self.needs_reset = True
            orig_ep_rew = sum(self.original_rewards)
            ep_info = {"original_r": orig_ep_rew}
            self.episode_orig_returns.append(orig_ep_rew)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"].update(ep_info)
        self.total_steps += 1
        return observation, reward, done, info

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.original_rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset()



def make_vec_env(
    env_name: str,
    *,
    rng: np.random.Generator,
    n_envs: int = 8,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
) -> VecEnv:
    """Makes a vectorized environment.

    Args:
        env_name: The Env's string id in Gym.
        rng: The random state to use to seed the environment.
        n_envs: The number of duplicate environments.
        parallel: If True, uses SubprocVecEnv; otherwise, DummyVecEnv.
        log_dir: If specified, saves Monitor output to this directory.
        max_episode_steps: If specified, wraps each env in a TimeLimit wrapper
            with this episode length. If not specified and `max_episode_steps`
            exists for this `env_name` in the Gym registry, uses the registry
            `max_episode_steps` for every TimeLimit wrapper (this automatic
            wrapper is the default behavior when calling `gym.make`). Otherwise
            the environments are passed into the VecEnv unwrapped.
        post_wrappers: If specified, iteratively wraps each environment with each
            of the wrappers specified in the sequence. The argument should be a Callable
            accepting two arguments, the Env to be wrapped and the environment index,
            and returning the wrapped Env.
        env_make_kwargs: The kwargs passed to `spec.make`.

    Returns:
        A VecEnv initialized with `n_envs` environments.
    """
    # Resolve the spec outside of the subprocess first, so that it is available to
    # subprocesses running `make_env` via automatic pickling.
    # Just to ensure packages are imported and spec is properly resolved
    tmp_env = gym.make(env_name)
    tmp_env.close()
    spec = tmp_env.spec
    env_make_kwargs = env_make_kwargs or {}

    def make_env(i: int, this_seed: int) -> gym.Env:
        # Previously, we directly called `gym.make(env_name)`, but running
        # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
        # created a weird interaction between Gym and Ray -- `gym.make` would fail
        # inside this function for any of our custom environment unless those
        # environments were also `gym.register()`ed inside `make_env`. Even
        # registering the custom environment in the scope of `make_vec_env` didn't
        # work. For more discussion and hypotheses on this issue see PR #160:
        # https://github.com/HumanCompatibleAI/imitation/pull/160.
        assert env_make_kwargs is not None  # Note: to satisfy mypy
        assert spec is not None  # Note: to satisfy mypy
        env = gym.make(spec, max_episode_steps=max_episode_steps, **env_make_kwargs)

        # Seed each environment with a different, non-sequential seed for diversity
        # (even if caller is passing us sequentially-assigned base seeds). int() is
        # necessary to work around gym bug where it chokes on numpy int64s.
        env.reset(seed=int(this_seed))
        # NOTE: we do it here rather than on the final VecEnv, because
        # that would set the same seed for all the environments.

        # Use Monitor to record statistics needed for Baselines algorithms logging
        # Optionally, save to disk
        log_path = None
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "monitor")
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.path.join(log_subdir, f"mon{i:03d}")

        env = CustomEnvMonitor(env, log_path)

        if post_wrappers:
            for wrapper in post_wrappers:
                env = wrapper(env, i)

        return env

    env_seeds = make_seeds(rng, n_envs)
    env_fns: List[Callable[[], gym.Env]] = [
        functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
    ]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        return DummyVecEnv(env_fns)