import json
import os
import re
import time

import gymnasium as gym
from imitation.data import types, rollout
from typing import List, Any, Mapping, Iterable, Sequence

import numpy as np
from imitation.data.types import stack_maybe_dictobs, AnyTensor
import torch.utils.data as th_data
import torch
import torch.nn as nn
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common import torch_layers
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize


class DictToArgs:
    def __init__(self, input_dict):
        for key in input_dict.keys():
            setattr(self, key, input_dict[key])


def import_wandb():
    try:
        import wandb
        from private_login import wandb_login
        wandb_login()
        return wandb
    except ImportError:
        return None


def flatten_trajectories(
        trajectories: Iterable[types.Trajectory],
) -> types.Transitions:
    """Flatten a series of trajectory dictionaries into arrays.

    Args:
        trajectories: list of trajectories.

    Returns:
        The trajectories flattened into a single batch of Transitions.
    """

    def all_of_type(key, desired_type):
        return all(
            isinstance(getattr(traj, key), desired_type) for traj in trajectories
        )

    assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)

    # mypy struggles without Any annotation here.
    # The necessary constraints are enforced above.
    keys = ["obs", "next_obs", "acts", "rews", "dones", "infos"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        parts["rews"].append(traj.rews)

        infos = np.array([{}] * len(traj))
        parts["infos"].append(infos)

    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.TransitionsWithRew(**cat_parts)


def transitions_with_rew_collate_fn(
        batch: Sequence[Mapping[str, np.ndarray]],
) -> Mapping[str, AnyTensor]:
    """Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.

    Use this as the `collate_fn` argument to `DataLoader` if using an instance of
    `TransitionsMinimal` as the `dataset` argument.

    Args:
        batch: The batch to collate.

    Returns:
        A collated batch. Uses Torch's default collate function for everything
        except the "infos" key. For "infos", we join all the info dicts into a
        list of dicts. (The default behavior would recursively collate every
        info dict into a single dict, which is incorrect.)
    """
    batch_acts_and_dones = [
        {k: np.array(v) for k, v in sample.items() if k in ["acts", "dones"]}
        for sample in batch
    ]

    result = th_data.dataloader.default_collate(batch_acts_and_dones)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample in batch]
    result["obs"] = stack_maybe_dictobs([sample["obs"] for sample in batch])
    result["next_obs"] = stack_maybe_dictobs([sample["next_obs"] for sample in batch])
    result["rews"] = stack_maybe_dictobs([sample["rews"] for sample in batch])
    return result


def load_json(file_json="your_file.json"):
    with open(file_json, "r") as file:
        data = json.load(file)
    return data


class StateDependentStdPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace SB3's fixed log_std with a network predicting log_std per state
        action_dim = self.action_dist.action_dim
        features_dim = kwargs["net_arch"][-1]
        self.log_std_net = nn.Linear(features_dim, action_dim)  # Network for log_std

        # Override the default log_std behavior
        # Remove SB3's existing log_std to avoid conflicts
        if hasattr(self, "log_std"):
            del self.log_std  # Explicitly remove the existing log_std
        # self.log_std = None  # Remove the fixed parameter
        self.register_buffer("log_std", torch.full((action_dim,), float("nan")))

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        log_std = self.log_std_net(latent_pi)  # Compute log_std per state
        # std = torch.exp(log_std)  # Convert to standard deviation
        return self.action_dist.proba_distribution(mean_actions, log_std)


def get_config(logdir, pathname="config.npy"):
    logdir = re.sub(r"model.*\.pth", "", logdir)
    return np.load(os.path.join(logdir, pathname), allow_pickle='TRUE').item()


def load_expert_transitions(env_name, n_envs, n_eval_episodes, n_expert_demos=50, seed=42, expert_algo="sac",
                            norm_reward=False):
    default_rng, env = load_env(env_name, n_envs, seed, norm_reward=norm_reward)
    expert = load_policy(
        f"{expert_algo}-huggingface",
        organization="HumanCompatibleAI",
        env_name=env_name,
        venv=env,
    )
    expert_rewards, _ = evaluate_policy(
        expert, env, n_eval_episodes, return_episode_rewards=True
    )
    target_rewards = np.mean(expert_rewards)
    print(f"Target:{target_rewards}")

    expert_rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=n_expert_demos),
        rng=default_rng,
        exclude_infos=False,
    )
    expert_transitions = rollout.flatten_trajectories_with_rew(expert_rollouts)
    return default_rng, env, expert_transitions, target_rewards


def load_env(env_name, n_envs, seed, env_make_kwargs=None, norm_reward=False, pre_wrappers=None):
    if pre_wrappers is None:
        pre_wrappers = []
    elif not isinstance(pre_wrappers, list):
        pre_wrappers = [lambda env, _: pre_wrappers(env)]
    else:
        pre_wrappers = [lambda env, _: wrapper(env) for wrapper in pre_wrappers]
    default_rng = np.random.default_rng(seed)
    env = make_vec_env(
        f"{env_name}",
        rng=default_rng,
        n_envs=n_envs,
        post_wrappers=pre_wrappers + [
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
        env_make_kwargs=env_make_kwargs
    )
    if norm_reward:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
    return default_rng, env


def get_policy_for(observation_space, action_space, net_arch):
    extractor = (
        torch_layers.CombinedExtractor
        if isinstance(observation_space, gym.spaces.Dict)
        else torch_layers.FlattenExtractor
    )
    if isinstance(action_space, gym.spaces.Discrete):
        return ActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            # Set lr_schedule to max value to force error if policy.optimizer
            # is used by mistake (should use self.optimizer instead).
            lr_schedule=lambda _: torch.finfo(torch.float32).max,
            features_extractor_class=extractor,
            net_arch=net_arch,
        )
    return StateDependentStdPolicy(
        observation_space=observation_space,
        action_space=action_space,
        # Set lr_schedule to max value to force error if policy.optimizer
        # is used by mistake (should use self.optimizer instead).
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
        features_extractor_class=extractor,
        net_arch=net_arch,
    )


class ObsCutWrapper(gym.Wrapper):
    """Cut the final observation - for hopper v3 vs v1 compatibility
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        sp = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=sp.low[:-1],
            high=sp.high[:-1],
            shape=(sp.shape[0] - 1,),
            dtype=sp.dtype,
        )

    def reset(self, **kwargs):
        new_obs, info = super().reset(**kwargs)
        return new_obs[..., :-1], info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs[..., :-1], rew, terminated, truncated, info


def create_logdir(env_name, seed):
    logdir = os.path.join('logs', 'train', env_name)
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{seed}'
    logdir = os.path.join(logdir, run_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
