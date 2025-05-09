import os
import re

import numpy as np
import torch
from imitation.algorithms import bc
from imitation.util import logger as imit_logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

import eirl2
import gflow
from CustomEnvMonitor import CartpoleVecEnvActionFlipWrapper
from callbacks import RewardLoggerCallback
# from CustomEnvMonitor import make_vec_env
from helper_local import import_wandb, load_expert_transitions, get_policy_for, create_logdir, init_policy_weights, get_target_rewards, \
    wrap_env_with_reward
from modified_cartpole import overridden_vec_env

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
wandb = import_wandb()

import eirl
from learner_configs import load_learner


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
              learner_timesteps=1000_000,
              gamma=0.995,
              training_increments=5,
              lr=0.0007172435323620212,
              l2_weight=0.,  # 1.3610189916104634e-6,
              batch_size=64,
              n_eval_episodes=50,
              n_envs=16,
              n_expert_demos=60,
              extra_tags=None,
              early_learning=False,
              env_name="seals:seals/Hopper-v1",
              overrides=None,
              expert_algo=None,
              override_env_name=None,
              enforce_rew_val_consistency=False,
              norm_reward=False,
              net_arch=None,
              rl_algo="ppo",
              reset_weights=False,
              rew_const=False,
              disc_coef=0.,
              flip_cartpole_actions=False,
              val_coef=1.,
              use_returns=True,
              use_z=True,
              kl_coef=1.,
              log_prob_loss=None,
              target_log_probs=False,
              use_scheduler=False,
              adv_coef=0.,
              target_back_probs=False,
              ):
    if flip_cartpole_actions and not re.search("CartPole", env_name):
        raise Exception(f"flip_cartpole_actions only works for CartPole envs")
    if net_arch is None:
        net_arch = [256, 256, 256, 256]
    if expert_algo is None:
        expert_algo = env_expert_algos[env_name]

    tags = [] + (extra_tags if extra_tags is not None else [])
    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), locals())
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=locals(), tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])
    default_rng, env, expert_transitions, target_rewards, expert, expert_rollouts = load_expert_transitions(env_name, n_envs, n_eval_episodes,
                                                                                   n_expert_demos, seed, expert_algo,
                                                                                   norm_reward)

    policy = get_policy_for(env.observation_space, env.action_space, net_arch)
    if model_file is not None:
        policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])

    if algo in ["eirl","eirl2"]:
        agent = None
        if algo == "eirl":
            agent = eirl.EIRL
        elif algo == "eirl2":
            agent = eirl2.EIRL
        expert_trainer = agent(
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
            rew_const=rew_const,
            disc_coef=disc_coef,
            #TODO: remove this unless you want loads of plots:!
            # logdir=logdir,
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
    elif algo == "gflow":
        expert_trainer = gflow.GFLOW(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_rollouts,
            gamma=gamma,
            batch_size=batch_size,
            l2_weight=l2_weight,
            rng=default_rng,
            custom_logger=custom_logger,
            net_arch=net_arch,
            reward_type=reward_type,
            val_coef=val_coef,
            hard=hard,
            use_returns=use_returns,
            use_z=use_z,
            kl_coef=kl_coef,
            log_prob_loss=log_prob_loss,
            target_log_probs=target_log_probs,
            use_scheduler=use_scheduler,
            n_epochs=n_epochs,
            adv_coef=adv_coef,
        )
    else:
        raise NotImplementedError(f"Unimplemented algorithm: {algo}")
    if model_file is not None:
        #TODO: load up reward networks too!
        raise NotImplementedError
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
        try:
            obj["reward_const"] = expert_trainer.reward_const.state_dict()
        except Exception as e:
            pass
        torch.save(obj, f'{logdir}/model_SUP_{epoch}.pth')
    if learner_timesteps == 0:
        wandb.finish()
        return
    env, wenv = override_env_and_wrap_reward(env, env_name, expert_trainer, log_prob_adj_reward, n_envs, neg_reward,
                                             override_env_name, overrides, reward_type)
    if flip_cartpole_actions:
        wenv = CartpoleVecEnvActionFlipWrapper(wenv)
    if overrides is not None:
        target_rewards = get_target_rewards(env, expert, n_eval_episodes)
        print(f"Expert Target Rewards in Overridden Environment:{target_rewards:.2f}")
    learner = load_learner(env_name, wenv, logdir, expert_trainer.policy, rl_algo)
    if reset_weights:
        learner.policy.apply(init_policy_weights)
    learner.learn(learner_timesteps, callback=RewardLoggerCallback())
    if flip_cartpole_actions:
        env = CartpoleVecEnvActionFlipWrapper(env)
    mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement", log=True)
    torch.save({'model_state_dict': learner.policy.state_dict()},
               f'{logdir}/model_RL_{learner_timesteps}.pth')
    # print(f"Timesteps:{learner_timesteps}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")
    wandb.finish()


def override_env_and_wrap_reward(env, env_name, expert_trainer, log_prob_adj_reward, n_envs, neg_reward,
                                 override_env_name, overrides, reward_type="state-action"):
    try:
        rew_const_adj = expert_trainer.reward_const.detach().cpu().numpy().item()
    except Exception:
        rew_const_adj = 0
    if log_prob_adj_reward:
        rfunc = expert_trainer.lp_adj_reward
    elif reward_type=="next state only":
        rfunc = expert_trainer.state_reward_func
    else:
        rfunc = expert_trainer.reward_func
    if overrides is not None:
        if override_env_name is None:
            override_env_name = env_name
        env = overridden_vec_env(override_env_name, n_envs, overrides)
    wenv = wrap_env_with_reward(env, rfunc, neg_reward, rew_const_adj, reward_type)
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


env_expert_algos = {
    "seals:seals/CartPole-v0": "ppo",
    "seals:seals/Hopper-v1": "sac",
    "seals:seals/Ant-v1": "ppo",
    "seals:seals/MountainCar-v0": "ppo",
    "seals:seals/Humanoid-v1": "ppo",
    "seals:seals/Walker2d-v0": "sac",
    "seals:seals/HalfCheetah-v0": "ppo",
}

if __name__ == "__main__":
    # logdir = "logs/train/seals:seals/Hopper-v1/2025-03-21__10-24-57__seed_0"
    # model_file = get_latest_model("logs/train/seals:seals/Hopper-v1/2025-03-21__10-24-57__seed_0", "SUP")
    for seed in [100, 0, 123, 412, 40, 32, 332, 32]:
        trainEIRL(
            algo="gflow",
            seed=seed,
            consistency_coef=50,
            n_expert_demos=2,
            n_eval_episodes=1,
            gamma=0.98,
            hard=True,
            norm_reward=True,
            disc_coef=0.,#0.01,
            l2_weight=0.0015,
            learner_timesteps=200_000,
            lr=6e-4,
            n_envs=32,
            n_epochs=10,
            rl_algo="ppo",
            model_file=None,
            reward_type="state",
            extra_tags=["disc"],
            env_name="seals:seals/CartPole-v0",
        )
