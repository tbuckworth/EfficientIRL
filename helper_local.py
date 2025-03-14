from imitation.data import types
from typing import List, Any, Mapping, Iterable, Sequence

import numpy as np
from imitation.data.types import stack_maybe_dictobs, AnyTensor
import torch.utils.data as th_data


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