import os
import time

import numpy as np
import torch
from imitation.data import rollout, wrappers
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards import reward_wrapper
from imitation.util import logger as imit_logger
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from callbacks import RewardLoggerCallback
# from CustomEnvMonitor import make_vec_env
from helper_local import import_wandb

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
wandb = import_wandb()

import eirl
from ant_v1_learner_config import load_ant_learner

SEED = 42  # Does this matter?


def wrap_env_with_reward(env, policy):
    def predict_processed(
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            done: np.ndarray,
            **kwargs,
    ) -> np.ndarray:
        # this is for the reward function signature
        with torch.no_grad():
            nobs = torch.FloatTensor(next_state).to(device=policy.device)
        return policy.predict_values(nobs).squeeze().detach().cpu().numpy()


    venv_buffering = wrappers.BufferingWrapper(env)
    venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
        venv_buffering,
        reward_fn=predict_processed,
    )
    return venv_wrapped


class WandbInfoLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:  # Ensure it's an episodic environment
            # Extract custom metric from the info dict
            infos = self.locals["infos"]
            ep_orig_rew = [info["episode"]["original_r"] for info in infos if "episode" in info.keys()]
            if ep_orig_rew != []:
                wandb.log({"rollout/ep_orig_rew_mean": np.mean(ep_orig_rew)}, step=self.num_timesteps)
        return True  # Continue training


def create_logdir(env_name, seed):
    logdir = os.path.join('logs', 'train', env_name)
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{seed}'
    logdir = os.path.join(logdir, run_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir

def main(consistency_coef=100.):
    ppo_timesteps = 0#1000_000
    env_name = "seals/Ant-v1"
    logdir = create_logdir(env_name, SEED)
    wandb.init(project="EfficientIRL", sync_tensorboard=True)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])
    training_increments = 5
    n_epochs = 100
    default_rng = np.random.default_rng(SEED)
    env = make_vec_env(
        f"seals:{env_name}",
        rng=default_rng,
        n_envs=8,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )

    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name=env_name,
        venv=env,
    )
    expert_rewards, _ = evaluate_policy(
        expert, env, 10, return_episode_rewards=True
    )
    target_rewards = np.mean(expert_rewards)
    print(f"Target:{target_rewards}")

    expert_rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=default_rng,
    )
    expert_transitions = rollout.flatten_trajectories(expert_rollouts)

    expert_trainer = eirl.EIRL(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_transitions,
        rng=default_rng,
        custom_logger=custom_logger,
        consistency_coef=consistency_coef,
    )

    for i, increment in enumerate([training_increments for i in range(n_epochs // training_increments)]):
        expert_trainer.train(n_epochs=increment,progress_bar=False)
        mean_rew, per_expert, std_err = evaluate(env, expert_trainer, target_rewards, phase="supervised",log=True)
        print(f"Epoch:{(i + 1) * increment}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")

    wenv = wrap_env_with_reward(env, expert_trainer.policy)
    # TODO: can i upload my agent somehow?
    learner = load_ant_learner(wenv, logdir)
    # for i in range(20):
    learner.learn(ppo_timesteps, callback=RewardLoggerCallback())
    mean_rew, per_expert, std_err = evaluate(env, expert_trainer, target_rewards, phase="reinforcement",log=True)
    print(f"Timesteps:{ppo_timesteps}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")


def evaluate(env, expert_trainer, target_rewards, phase, log=False):
    rewards, _ = evaluate_policy(
        expert_trainer.policy, env, 10, return_episode_rewards=True
    )
    mean_rew = np.mean(rewards)
    std_err = np.std(rewards) / np.sqrt(len(rewards))
    per_expert = mean_rew / target_rewards
    if log:
        wandb.log({
            "reward/original_ep_return_mean": mean_rew,
            "reward/original_ep_return_std_err": std_err,
            "reward/original_ep_return_ratio": per_expert,
        })
    return mean_rew, per_expert, std_err


if __name__ == "__main__":
    main()
