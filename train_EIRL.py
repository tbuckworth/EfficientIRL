import os
import time

import numpy as np
import torch
from imitation.algorithms import bc
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
from helper_local import import_wandb, flatten_trajectories

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
wandb = import_wandb()

import eirl
from ant_v1_learner_config import load_ant_ppo_learner, load_ant_sac_learner, load_hopper_ppo_learner, load_ppo_learner


def wrap_env_with_reward(env, reward_func, neg_reward=False, rew_const_adj=0., ):
    def predict_processed(
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray = None,
            done: np.ndarray = None,
            **kwargs,
    ) -> np.ndarray:
        # this is for the reward function signature

        with torch.no_grad():
            obs = torch.FloatTensor(state).to(device=reward_func.device)
            acts = torch.FloatTensor(action).to(device=reward_func.device)
            rew = reward_func(obs, acts, None, None).squeeze().detach().cpu().numpy()
            if neg_reward:
                return -rew
            return rew + rew_const_adj

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


def main(algo="eirl", seed=42, hard=True,
         consistency_coef=100., n_epochs=20):
    use_next_state_reward = True
    neg_reward = False
    rew_const_adj = 0
    learner_timesteps = 1000_000
    gamma = 0.995
    training_increments = 5
    lr = 0.0007172435323620212
    l2_weight = 0  # 1.3610189916104634e-6
    batch_size = 64
    n_eval_episodes = 50
    n_envs = 16
    n_expert_demos = 60

    env_name = "seals/Hopper-v1"

    tags = ["HopperComp", "Fixed Entropy", "PPO", "NEXT STATE BASED"]
    logdir = create_logdir(env_name, seed)

    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=locals(), tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])
    default_rng, env, expert_transitions, target_rewards = load_expert_transitions(env_name, n_envs, n_eval_episodes,
                                                                                   n_expert_demos, seed)

    if algo == "eirl":
        expert_trainer = eirl.EIRL(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=default_rng,
            custom_logger=custom_logger,
            consistency_coef=consistency_coef,
            hard=hard,
            gamma=gamma,
            batch_size=batch_size,
            l2_weight=l2_weight,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": lr},
            use_next_state_reward=use_next_state_reward,
        )
    elif algo == "bc":
        expert_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=default_rng,
            custom_logger=custom_logger,
            batch_size=batch_size,
            l2_weight=l2_weight,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": lr},
        )
    else:
        raise NotImplementedError(f"Unimplemented algorithm: {algo}")

    for i, increment in enumerate([training_increments for i in range(n_epochs // training_increments)]):
        expert_trainer.train(n_epochs=increment, progress_bar=False)
        mean_rew, per_expert, std_err = evaluate(env, expert_trainer, target_rewards, phase="supervised", log=True)
        print(f"Epoch:{(i + 1) * increment}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")
    if learner_timesteps == 0:
        wandb.finish()
        return

    wenv = wrap_env_with_reward(env, expert_trainer.reward_func, neg_reward, rew_const_adj)
    learner = load_ppo_learner(env_name, wenv, logdir, expert_trainer.policy)
    # for i in range(20):
    learner.learn(learner_timesteps, callback=RewardLoggerCallback())
    mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement",log=True)
    # print(f"Timesteps:{learner_timesteps}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")
    wandb.finish()


def load_expert_transitions(env_name, n_envs, n_eval_episodes, n_expert_demos=50, seed=42):
    default_rng = np.random.default_rng(seed)
    env = make_vec_env(
        f"seals:{env_name}",
        rng=default_rng,
        n_envs=n_envs,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )
    expert = load_policy(
        "sac-huggingface",
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
    for algo in ["eirl"]:
        for n_epochs in [50]:
            for seed in [0, 100, 123, 412]:  # , 352, 342, 3232, 23243, 233343]:
                for hard in [False, True]:
                    for consistency_coef in [100.]:
                        main(algo, seed, hard=hard, n_epochs=n_epochs, consistency_coef=consistency_coef)
