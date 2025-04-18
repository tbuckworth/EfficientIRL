import os

import numpy as np
import torch
from imitation.util import logger as imit_logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

import imeow
from callbacks import RewardLoggerCallback
# from CustomEnvMonitor import make_vec_env
from helper_local import import_wandb, load_expert_transitions, create_logdir, get_latest_model, \
    get_config, load_reward_models, wrap_env_with_reward, \
    create_envs_meow_imitation_compat
from meow.meow_continuous_action import FlowPolicy, MEOW
from modified_cartpole import overridden_vec_env

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
wandb = import_wandb()

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


def trainIMEow(algo="imeow",
               seed=42,
               consistency_coef=1.,
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
               l2_weight=0,  # 1.3610189916104634e-6,
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
               alpha=0.2,
               sigma_max=-0.3,
               sigma_min=-5.0,
               q_coef=1.,
               hard=False,
               abs_log_probs=False,
               convex_opt=False,
               calc_log_probs=False,
               ):
    if net_arch is None:
        net_arch = [256, 256, 256, 256]
    if expert_algo is None:
        expert_algo = env_expert_algos[env_name]

    tags = [] + (extra_tags if extra_tags is not None else [])
    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), locals())
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=locals(), tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])
    default_rng, env, expert_transitions, target_rewards, expert, rollouts = load_expert_transitions(env_name, n_envs,
                                                                                           n_eval_episodes,
                                                                                           n_expert_demos, seed,
                                                                                           expert_algo,
                                                                                           norm_reward)
    policy = None
    cfg = {}
    if model_file is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = get_config(model_file)
        policy = FlowPolicy(alpha=cfg["alpha"],
                            sigma_max=cfg["sigma_max"],
                            sigma_min=cfg["sigma_min"],
                            action_sizes=env.action_space.shape[-1],
                            state_sizes=env.observation_space.shape[-1],
                            device=device).to(device)
        policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])

    if algo == "imeow":
        expert_trainer = imeow.IMEow(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=default_rng,
            custom_logger=custom_logger,
            consistency_coef=consistency_coef,
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
            alpha=alpha,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            q_coef=q_coef,
            hard=hard,
            abs_log_probs=abs_log_probs,
            convex_opt=convex_opt,
            calc_log_probs=calc_log_probs,
        )
    else:
        raise NotImplementedError(f"Unimplemented algorithm: {algo}")
    if model_file is not None:
        load_reward_models(cfg, expert_trainer, model_file, policy)

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

    # envs, test_envs = create_envs_meow(env_name, seed, n_envs)
    envs, test_envs = create_envs_meow_imitation_compat(env_name, n_envs, norm_reward, seed)

    env, wenv = override_env_and_wrap_reward(envs, env_name, expert_trainer, log_prob_adj_reward, n_envs, neg_reward,
                                             override_env_name, overrides)
    if rl_algo == "meow":
        learner = MEOW(
            wenv,
            test_envs,  # these not overridden!
            policy=expert_trainer.policy,
            logdir=logdir,
            evaluate=lambda e, l: evaluate(e, l, target_rewards, phase="reinforcement", log=True),
        )
    else:
        learner = load_learner(env_name, wenv, logdir, None, rl_algo)
    learner.learn(learner_timesteps, wandb=wandb, callback=RewardLoggerCallback())
    mean_rew, per_expert, std_err = evaluate(env, learner, target_rewards, phase="reinforcement", log=True)
    torch.save({'model_state_dict': learner.policy.state_dict()},
               f'{logdir}/model_RL_{learner_timesteps}.pth')
    wandb.finish()


def override_env_and_wrap_reward(env, env_name, expert_trainer, log_prob_adj_reward, n_envs, neg_reward,
                                 override_env_name, overrides):
    try:
        rew_const_adj = expert_trainer.reward_const.detach().cpu().numpy().item()
    except Exception:
        rew_const_adj = 0
    if log_prob_adj_reward:
        rfunc = expert_trainer.lp_adj_reward
    else:
        rfunc = expert_trainer.reward_func
    if overrides is not None:
        if override_env_name is None:
            override_env_name = env_name
        env = overridden_vec_env(override_env_name, n_envs, overrides)
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


env_expert_algos = {
    "seals:seals/Cartpole-v0": "ppo",
    "seals:seals/Hopper-v1": "sac",
    "seals:seals/Ant-v1": "ppo",
    "seals:seals/MountainCar-v0": "ppo",
    "seals:seals/Humanoid-v1": "ppo",
    "seals:seals/Walker2d-v0": "sac",
    "seals:seals/HalfCheetah-v0": "ppo",
    "seals:seals/Pendulum-v1": "ppo", #Use absorb wrapper?
    "seals:seals/Swimmer-v1": "ppo",
}

if __name__ == "__main__":
    # logdir = "logs/train/seals:seals/Hopper-v1/2025-03-21__10-24-57__seed_0"
    model_file = get_latest_model("logs/train/seals:seals/Hopper-v1/2025-03-26__12-55-29__seed_0", "SUP")
    for seed in [100, 0, 123, 412, 40, 32, 332, 32]:
        trainIMEow(
            algo="imeow",
            seed=seed,
            consistency_coef=0.,
            q_coef=1.,
            n_expert_demos=1,
            n_eval_episodes=1,
            rl_algo="meow",
            model_file=model_file,
            n_epochs=150,
            reward_type="next state",
            extra_tags=["meow"],
            learner_timesteps=3000_000,
            env_name="seals:seals/Hopper-v1",
            #TODO: try these out, maybe use pendulum
            convex_opt=False,
            calc_log_probs=False,
        )
