import json

from imitation.data import types
from typing import List, Any, Mapping, Iterable, Sequence

import numpy as np
from imitation.data.types import stack_maybe_dictobs, AnyTensor
import torch.utils.data as th_data
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

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
        action_dim = self.action_dist.proba_distribution._param_shape[0]

        self.log_std_net = nn.Linear(self.features_dim, action_dim)  # Network for log_std

        # Override the default log_std behavior
        self.log_std = None  # Remove the fixed parameter

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        log_std = self.log_std_net(latent_pi)  # Compute log_std per state
        std = torch.exp(log_std)  # Convert to standard deviation
        return self.action_dist.proba_distribution(mean_actions, std)