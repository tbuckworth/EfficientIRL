"""Trains DAgger on synthetic demonstrations generated from an expert policy."""

import logging
import os.path as osp
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy as np
from sacred.observers import FileStorageObserver

from imitation.algorithms import dagger as dagger_algorithm
from imitation.algorithms import sqil as sqil_algorithm
from imitation.data import rollout, types
from imitation.scripts.ingredients import bc as bc_ingredient
from imitation.scripts.ingredients import demonstrations, environment, expert
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation
from imitation.util import util

from hp_tune.train_eirl_config import train_eirl_ex

logger = logging.getLogger(__name__)


def _all_trajectories_have_reward(trajectories: Sequence[types.Trajectory]) -> bool:
    """Returns True if all trajectories have reward information."""
    return all(isinstance(t, types.TrajectoryWithRew) for t in trajectories)


def _try_computing_expert_stats(
    expert_trajs: Sequence[types.Trajectory],
) -> Optional[Mapping[str, float]]:
    """Adds expert statistics to `stats` if all expert trajectories have reward."""
    if _all_trajectories_have_reward(expert_trajs):
        return rollout.rollout_stats(
            cast(Sequence[types.TrajectoryWithRew], expert_trajs),
        )
    else:
        logger.warning(
            "Expert trajectories do not have reward information, so expert "
            "statistics cannot be computed.",
        )
        return None


def _collect_stats(
    imit_stats: Mapping[str, float],
    expert_trajs: Sequence[types.Trajectory],
) -> Mapping[str, Mapping[str, Any]]:
    stats = {"imit_stats": imit_stats}
    expert_stats = _try_computing_expert_stats(expert_trajs)
    if expert_stats is not None:
        stats["expert_stats"] = expert_stats

    return stats


@train_eirl_ex.command
def eirl(
    eirl: Dict[str, Any],
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    """Runs EIRL training.

    Args:
        eirl: Configuration for EIRL training.
        _run: Sacred run object.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    custom_logger, log_dir = logging_ingredient.setup_logging()

    expert_trajs = demonstrations.get_expert_trajectories()
    with environment.make_venv() as venv:  # type: ignore[wrong-arg-count]
        eirl_trainer = bc_ingredient.make_bc(venv, expert_trajs, custom_logger)

        eirl_train_kwargs = dict(log_rollouts_venv=venv, **eirl["train_kwargs"])
        if eirl_train_kwargs["n_epochs"] is None and eirl_train_kwargs["n_batches"] is None:
            eirl_train_kwargs["n_batches"] = 50_000

        eirl_trainer.train(**eirl_train_kwargs)
        # TODO(adam): add checkpointing to BC?
        util.save_policy(eirl_trainer.policy, policy_path=osp.join(log_dir, "final.th"))

        imit_stats = policy_evaluation.eval_policy(eirl_trainer.policy, venv)

    stats = _collect_stats(imit_stats, expert_trajs)

    return stats


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_imitation"
    observer = FileStorageObserver(observer_path)
    train_eirl_ex.observers.append(observer)
    train_eirl_ex.run_commandline()


if __name__ == "__main__":
    main_console()
