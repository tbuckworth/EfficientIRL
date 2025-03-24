import os

import gymnasium as gym
import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data import wrappers
from imitation.rewards import reward_wrapper
from imitation.util import logger as imit_logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from callbacks import RewardLoggerCallback
# from CustomEnvMonitor import make_vec_env
from helper_local import import_wandb, load_expert_transitions, get_policy_for, create_logdir, get_latest_model
from modified_cartpole import overridden_vec_env

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
wandb = import_wandb()

import eirl
from ant_v1_learner_config import load_learner


def wrap_env_with_reward(env, reward_func, neg_reward=False, rew_const_adj=0., ):
    n_actions = None
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if is_discrete:
        n_actions = env.action_space.n

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
            if is_discrete:
                acts = torch.nn.functional.one_hot(torch.LongTensor(action), n_actions).to(device=reward_func.device)
            else:
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


def trainEIRL(algo="eirl",
              seed=42,
              hard=False,
              consistency_coef=100.,
              n_epochs=20,
              model_file=None,
              reward_type="next state",
              log_prob_adj_reward=False,
              neg_reward=False,
              maximize_reward=False,
              rew_const_adj=0,
              learner_timesteps=1000_000,
              gamma=0.995,
              training_increments=5,
              lr=0.0007172435323620212,
              l2_weight=0,  # 1.3610189916104634e-6,
              batch_size=64,
              n_eval_episodes=50,
              n_envs=16,
              n_expert_demos=60,
              extra_tags=None,
              early_learning=False,
              env_name="seals:seals/Hopper-v1",
              overrides=None,
              expert_algo="sac",
              override_env_name=None,
              enforce_rew_val_consistency=True,
              norm_reward=True,
              net_arch=None,
              rl_algo="ppo",
              ):
    if net_arch is None:
        net_arch = [256, 256, 256, 256]

    tags = [] + (extra_tags if extra_tags is not None else [])
    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), locals())
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=locals(), tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])
    default_rng, env, expert_transitions, target_rewards = load_expert_transitions(env_name, n_envs, n_eval_episodes,
                                                                                   n_expert_demos, seed, expert_algo,
                                                                                   norm_reward)

    policy = get_policy_for(env.observation_space, env.action_space, net_arch)
    if model_file is not None:
        policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])

    if algo == "eirl":
        expert_trainer = eirl.EIRL(
            policy=policy,
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
            reward_type=reward_type,
            maximize_reward=maximize_reward,
            log_prob_adj_reward=log_prob_adj_reward,
            enforce_rew_val_consistency=enforce_rew_val_consistency,
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
    epoch = None
    for i, increment in enumerate([training_increments for i in range(n_epochs // training_increments)]):
        expert_trainer.train(n_epochs=increment, progress_bar=False)
        mean_rew, per_expert, std_err = evaluate(env, expert_trainer, target_rewards, phase="supervised", log=True)
        epoch = (i + 1) * increment
        print(f"Epoch:{epoch}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")
        if per_expert > 1 and early_learning:
            break
    if epoch is not None:
        obj = {'model_state_dict': expert_trainer.policy.state_dict()}
        obj["reward_func"] = expert_trainer.reward_func.state_dict()
        if reward_type == "next state":
            obj["state_reward_func"] = expert_trainer.state_reward_func.state_dict()
        if log_prob_adj_reward:
            obj["lp_adj_reward"] = expert_trainer.lp_adj_reward.state_dict()

        torch.save(obj, f'{logdir}/model_SUP_{epoch}.pth')
    if learner_timesteps == 0:
        wandb.finish()
        return
    env, wenv = override_env_and_wrap_reward(env, env_name, expert_trainer, log_prob_adj_reward, n_envs, neg_reward,
                                             override_env_name, overrides, rew_const_adj)
    learner = load_learner(env_name, wenv, logdir, expert_trainer.policy, rl_algo)
    learner.learn(learner_timesteps, callback=RewardLoggerCallback())
    mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement", log=True)
    torch.save({'model_state_dict': learner.policy.state_dict()},
               f'{logdir}/model_RL_{learner_timesteps}.pth')
    # print(f"Timesteps:{learner_timesteps}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")
    wandb.finish()


def override_env_and_wrap_reward(env, env_name, expert_trainer, log_prob_adj_reward, n_envs, neg_reward,
                                 override_env_name, overrides, rew_const_adj):
    if log_prob_adj_reward:
        rfunc = expert_trainer.lp_adj_reward
    else:
        rfunc = expert_trainer.reward_func
    if overrides is not None:
        if override_env_name is None:
            override_env_name = env_name
        env = overridden_vec_env(override_env_name, n_envs, overrides)
        wenv = wrap_env_with_reward(env, rfunc, neg_reward, rew_const_adj)
    else:
        wenv = wrap_env_with_reward(env, rfunc, neg_reward, rew_const_adj)
    return env, wenv


def evaluate(env, expert_trainer, target_rewards, phase, log=False, callback=None):
    rewards, _ = evaluate_policy(
        expert_trainer.policy, env, 10, return_episode_rewards=True, callback=callback
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


env_names = [
    "seals:seals/Cartpole-v0",
    "seals:seals/Hopper-v1",
    "seals:seals/Ant-v1",
    "seals:seals/MountainCar-v0",
]

if __name__ == "__main__":
    model_file = get_latest_model("logs/train/seals:seals/Ant-v1/2025-03-22__05-15-35__seed_0", "SUP")
    for algo in ["eirl"]:
        for n_epochs in [0]:
            for maximize_reward in [False]:  # , True]:
                for hard in [False]:  # , True]:
                    for enforce_rew_val_consistency in [False]:
                        for seed in [100, 0, 123, 412, 40, 32, 332, 32]:
                            for reward_type in ["next state"]:  # , "state-action", "next state", "state"]:
                                trainEIRL(
                                    algo, seed,
                                    n_expert_demos=1,
                                    rl_algo="sac",
                                    model_file=model_file,
                                    n_epochs=n_epochs,
                                    reward_type=reward_type,
                                    maximize_reward=maximize_reward,
                                    extra_tags=["ant"],
                                    early_learning=False,
                                    learner_timesteps=3000_000,
                                    env_name="seals:seals/Ant-v1",
                                    override_env_name=None,  # "MountainCar-v0",
                                    overrides=None,  # {"gravity": 15.0},
                                    expert_algo="ppo",
                                    hard=hard,
                                    enforce_rew_val_consistency=enforce_rew_val_consistency,
                                    norm_reward=False,
                                    # n_expert_demos=1,
                                )
