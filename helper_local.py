import json
import os
import re
import subprocess
import time

import gymnasium as gym
from imitation.data import types, rollout, wrappers
from typing import List, Any, Mapping, Iterable, Sequence

import numpy as np
from imitation.data.types import stack_maybe_dictobs, AnyTensor
import torch.utils.data as th_data
import torch
import torch.nn as nn
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards import reward_wrapper
from imitation.util.util import make_vec_env
from stable_baselines3 import TD3, PPO
from stable_baselines3.common import torch_layers
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize

try:
    import private_login
except ImportError:
    pass

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
    default_rng, env, expert_rollouts, target_rewards, expert = load_expert_rollouts(env_name, expert_algo, n_envs,
                                                                             n_eval_episodes, n_expert_demos,
                                                                             norm_reward, seed)
    expert_transitions = rollout.flatten_trajectories_with_rew(expert_rollouts)
    return default_rng, env, expert_transitions, target_rewards, expert


def load_cartpole_ppo_expert(env):
    # Deliberately sub_optimal agent - gets around 150 reward
    logdir = "logs/train/seals:seals/CartPole-v0/2025-03-26__07-55-45__seed_42"
    return load_agent(env, PPO, logdir)


def load_expert_rollouts(env_name, expert_algo, n_envs, n_eval_episodes, n_expert_demos, norm_reward, seed):
    default_rng, env = load_env(env_name, n_envs, seed, norm_reward=norm_reward)
    if expert_algo == "td3" and env_name == "seals:seals/Hopper-v1":
        expert = load_hopper_td3_expert(env)
    elif expert_algo == "ppo" and env_name == "seals:seals/CartPole-v0":
        expert = load_cartpole_ppo_expert(env)
    else:
        expert = load_policy(
            f"{expert_algo}-huggingface",
            organization="HumanCompatibleAI",
            env_name=env_name,
            venv=env,
        )
    target_rewards = get_target_rewards(env, expert, n_eval_episodes)
    print(f"Target:{target_rewards}")
    expert_rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=n_expert_demos),
        rng=default_rng,
        exclude_infos=False,
    )
    return default_rng, env, expert_rollouts, target_rewards, expert


def get_target_rewards(env, expert, n_eval_episodes):
    expert_rewards, _ = evaluate_policy(
        expert, env, n_eval_episodes, return_episode_rewards=True
    )
    target_rewards = np.mean(expert_rewards)
    return target_rewards


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


def scp_request_pword(remote_dir, folder):
    import getpass
    import pexpect
    import sys
    # rsa_password = getpass.getpass("Enter RSA passphrase: ")
    rsa_password = private_login.rsa_passphrase
    # Spawn the scp process
    child = pexpect.spawn(f"scp -r {remote_dir} {folder}", encoding='utf-8')
    # Look for the password prompt and then send the password
    child.logfile = sys.stdout
    # Look for a passphrase prompt, EOF, or timeout
    index = child.expect([r"(?i)passphrase", pexpect.EOF, pexpect.TIMEOUT])
    if index == 0:
        child.sendline(rsa_password)
        child.expect(pexpect.EOF)
    elif index == 1:
        # Process finished without asking for a passphrase
        pass
    else:
        raise TimeoutError("Timed out waiting for the passphrase prompt.")


def get_latest_model(folder, keyword):
    if not os.path.exists(folder) or len(os.listdir(folder)) == 0:
        remote_dir = os.path.join("tfb115@shell1.doc.ic.ac.uk:/vol/bitbucket/tfb115/EfficientIRL", folder)
        sup_folder = os.path.dirname(folder)
        if not os.path.exists(sup_folder):
            os.makedirs(sup_folder)
        scp_request_pword(remote_dir, sup_folder)
        # process = subprocess.Popen(["scp", "-r", remote_dir, folder])
        # exit_code = process.wait()  # This will block until the process finishes
        # if exit_code != 0:
        #     scp_command = f"scp -r {remote_dir} {folder}"
        #     raise FileNotFoundError(f"Model not found. Consider Running:\n{scp_command}")
    search = lambda x: re.search(rf"model_{keyword}_(\d*).pth", x)
    if search(folder):
        return folder
    files = [os.path.join(folder, x) for x in os.listdir(folder)]
    last_checkpoint = max([int(search(x).group(1)) for x in files if search(x)])
    return [x for x in files if re.search(f"model_{keyword}_{last_checkpoint}.pth", x)][0]


def filter_params(params, function):
    import inspect
    acceptable_params = inspect.signature(function).parameters
    filtered_params = {k: v for k, v in params.items() if k in acceptable_params}
    return filtered_params

def init_policy_weights(m):
    """Mimics SB3’s default policy init: orthogonal for weights, zero for biases."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1.0)
        nn.init.constant_(m.bias.data, 0.0)


def load_agent(env, agent, logdir):
    model_file = get_latest_model(logdir, "RL")
    learner = agent('MlpPolicy', env)
    policy = learner.policy
    policy.load_state_dict(torch.load(model_file, map_location=policy.device)["model_state_dict"])
    return policy

def load_hopper_td3_expert(env):
    return load_agent(env, TD3, "logs/train/seals:seals/Hopper-v1/2025-03-21__17-06-20__seed_42")


def load_reward_models(cfg, expert_trainer, model_file, policy):
    expert_trainer.reward_func.load_state_dict(
        torch.load(model_file, map_location=policy.device
                   )["reward_func"])
    if cfg["reward_type"] == "next state":
        expert_trainer.state_reward_func.load_state_dict(
            torch.load(model_file, map_location=policy.device
                       )["state_reward_func"])
    if cfg["log_prob_adj_reward"]:
        expert_trainer.lp_adj_reward.load_state_dict(
            torch.load(model_file, map_location=policy.device
                       )["lp_adj_reward"])
    try:
        expert_trainer.reward_const.load_state_dict(
            torch.load(model_file, map_location=policy.device
                       )["reward_const"])
    except Exception as e:
        pass


def wrap_env_with_reward(env, reward_func, neg_reward=False, rew_const_adj=0., reward_type="state-action"):
    n_actions = None
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if is_discrete:
        n_actions = env.action_space.n

    if reward_type == "next state only":
        def predict_processed(
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray = None,
                done: np.ndarray = None,
                **kwargs,
        ) -> np.ndarray:
            # this is for the reward function signature
            with torch.no_grad():
                # obs = torch.FloatTensor(state).to(device=reward_func.device)
                obs = torch.FloatTensor(state).to(device=reward_func.device)
                rew = reward_func(obs, None, obs, None).squeeze().detach().cpu().numpy()
                if neg_reward:
                    return -rew
                return rew + rew_const_adj
    else:
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
