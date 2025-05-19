import threading
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import gflow
from helper_local import DictToArgs, filter_params
from tabular import TabularMDP, AscenderLong, OneStep, MattGridworld, CustMDP, DogSatMat
import gymnasium
import concurrent.futures
import traceback


def significance_matrix(df, mean_col, std_col):
    from scipy.stats import norm

    # Assuming your dataframe is called df and has columns 'correls_mean' and 'correls_std'
    means = df[mean_col].values
    stds = df[std_col].values

    # Create difference and standard error matrices via broadcasting
    diff = means[:, None] - means[None, :]  # Difference between means (n x n matrix)
    se = np.sqrt(stds[:, None] ** 2 + stds[None, :] ** 2)  # Standard error assuming independence

    # Compute t-statistics for a one-tailed test (row mean > column mean)
    t_stat = diff / se

    # For a one-sided test, the p-value is given by: p = 1 - CDF(t_stat)
    p_matrix = 1 - norm.cdf(t_stat)

    # Boolean matrix: True if row has significantly higher mean than column at p < 0.05
    signif_matrix = p_matrix < 0.05

    # Optional: convert to pandas DataFrame for readability
    p_values_df = pd.DataFrame(p_matrix, index=df.index, columns=df.index)
    signif_df = pd.DataFrame(signif_matrix, index=df.index, columns=df.index)

    print("P-values matrix:")
    print(p_values_df)
    print("\nSignificance matrix (True if row mean > column mean significantly at p < 0.05):")
    print(signif_df)


class NonTabularMDP:
    def __init__(self, tabular_mdp: TabularMDP, horizon: int, device=None):
        assert horizon > 1, "horizon must be greater than 1"
        self.horizon = horizon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.env = tabular_mdp
        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions
        self.mu = self.env.mu
        self.T = self.env.T
        self.R = self.env.reward_vector
        self.observation_space = gymnasium.spaces.Box(0, 1, (self.n_states,))
        self.action_space = gymnasium.spaces.Discrete(self.n_actions)

    def generate_trajectories(self, policy_name: str, n_traj: int, temp: int) -> List[torch.tensor]:
        assert policy_name in self.env.policies.keys(), f"Policy {policy_name} not found in Env {self.env.name}."
        pi = self.env.policies[policy_name].pi
        pi = (pi.log() / temp).softmax(dim=-1)
        output = []
        for i in range(n_traj):
            # sample state from mu:
            state_idx = self.sample(self.mu)
            obs = self.get_state(state_idx)
            # obs = s0.unsqueeze(0)
            acts = []
            rews = []
            for n in range(self.horizon - 1):
                action_probs = pi[state_idx].squeeze()
                action = self.sample(action_probs)
                next_state_probs = self.T[state_idx, action].squeeze()
                state_idx = self.sample(next_state_probs)
                reward = self.R[state_idx]
                s = self.get_state(state_idx)
                obs = torch.cat((obs, s), dim=0)
                acts.append(action.item())
                rews.append(reward.item())
            traj = DictToArgs(dict(
                obs=obs.cpu().numpy(),
                acts=np.array(acts),
                rews=np.array(rews),
                infos={},
                terminal=True,  # ...or actually calc this in mdps?
            ))
            output.append(traj)
        return output

    @staticmethod
    def sample(x: torch.tensor):
        assert x.sum() >= 1 - 1e-5, "x is not a prob dist"
        r = np.random.rand()
        return (x.cumsum(dim=-1) > r).argwhere()[0]

    def get_state(self, i):
        if isinstance(i, torch.Tensor):
            t = i
        else:
            t = torch.tensor([i])
        return torch.nn.functional.one_hot(t, self.n_states).to(self.device)


def get_idx(i, lens):
    assert i < np.prod(lens), f"i must be below {np.prod(lens)}, not {i}"
    output = []
    for l in lens:
        output.append(i % l)
        i = i // l
    return output


def run_experiment(n_threads=8):
    # ranges = dict(
    #     gamma=[0.99],
    #     net_arch=[[8, 8]],
    #     log_prob_loss=["kl"],
    #     target_log_probs=[True],
    #     target_back_probs=[True, False],
    #     reward_type=["next state only", "state"],
    #     adv_coef=[0, 1.],
    #     horizon=[7],
    #     n_epochs=[100],
    #     policy_name=["Hard Smax"],
    #     n_traj=[20, 100],
    #     temp=[1, 5],
    #     n_trials=[5],
    #     n_states=[6],
    #     lr=[1e-3],
    #     val_coef=[0, 0.5],
    #     hard=[True, False],
    #     use_returns=[True, False],
    #     use_z=[True, False],
    #     kl_coef=[1.],
    #     use_scheduler=[False],
    #     env_cons=AscenderLong,
    # )
    # test ranges:
    ranges = dict(
        gamma=[0.99],
        net_arch=[[64, 64, 64]],
        log_prob_loss=["kl"],
        target_log_probs=[True],
        target_back_probs=[True],
        reward_type=["next state only"],
        adv_coef=[0.],
        horizon=[3, 4, 5],
        n_epochs=[100],
        policy_name=["Hard Smax"],
        n_traj=[10],
        temp=[1, 3],
        n_trials=[5],
        n_states=[6],
        lr=[1e-3],
        val_coef=[0],
        hard=[True],
        use_returns=[True],
        use_z=[True],
        kl_coef=[1.],
        use_scheduler=[False],
        split_training=[0.3],
        value_is_potential=[False],
        env_cons=[DogSatMat],  # [OneStep, AscenderLong, MattGridworld, CustMDP, DogSatMat],
    )
    lens = [len(v) for k, v in ranges.items()]
    n_experiments = np.prod(lens)
    print(f"Running {n_experiments} experiments across {n_threads} threads")

    def subfunction(i):
        # Process each experiment in the sublist and return results
        # results = []
        idx = get_idx(i, lens)
        cfg = {k: ranges[k][i] for k, i in zip(ranges.keys(), idx)}
        # Simulate some processing and return a result (modify as needed)
        # results.append(run_config(cfg))
        return run_config(cfg)

    all_results = run_concurrently(n_experiments, n_threads, subfunction)
    df = pd.DataFrame(all_results)
    run_name = f"data/experiments_{time.strftime("%Y-%m-%d__%H-%M-%S")}.csv"
    df.to_csv(run_name, index=False)
    return


def split_list(n_experiments, n_threads):
    experiments = list(range(n_experiments))
    chunk_size = (n_experiments + n_threads - 1) // n_threads  # Ceiling division
    return [experiments[i:i + chunk_size] for i in range(0, n_experiments, chunk_size)]


def run_concurrently(n_experiments, n_threads, subfunction):
    # chunks = split_list(n_experiments, n_threads)
    results = []

    progress = tqdm(total=n_experiments, desc="Processing experiments")
    lock = threading.Lock()

    # Using ThreadPoolExecutor to handle threads and capture return values
    # with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
    #     futures = [executor.submit(subfunction, chunk) for chunk in chunks]
    #     for future in concurrent.futures.as_completed(futures):
    #         collated_results.extend(future.result())

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Map each future to the number of experiments it processes
        futures = [executor.submit(subfunction, exp) for exp in range(n_experiments)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            # Update progress by number of experiments processed in this chunk
            with lock:
                progress.update(1)
    progress.close()
    return results


def run(env_cons=AscenderLong, ):
    cfg = dict(
        gamma=0.99,
        net_arch=[8, 8],
        log_prob_loss="kl",
        target_log_probs=True,
        target_back_probs=True,
        reward_type="next state only",
        adv_coef=1.,
        horizon=20,
        n_epochs=100,
        policy_name="Hard Smax",
        n_traj=100,
        temp=5,
        n_trials=1,
        n_states=6,
        lr=0.001,
        val_coef=0.0,
        hard=False,
        use_returns=True,
        use_z=True,
        kl_coef=1.0,
        use_scheduler=False,
        verbose=True,
        env_cons=env_cons,
    )
    results = run_config(cfg)


def run_config(cfg):
    horizon = cfg["horizon"]
    n_trials = cfg["n_trials"]
    policy_name = cfg["policy_name"]
    env_cons = cfg["env_cons"]

    env_args = filter_params(cfg, env_cons)

    env = env_cons(**env_args)
    nt_env = NonTabularMDP(env, horizon)
    target_ret = env.evaluate_policy(nt_env.env.policies[policy_name])
    scores = []
    correls = []
    for i in range(n_trials):
        try:
            exp_trainer = run_gflow(cfg, nt_env)
        except Exception as e:
            print(f"Error in trial {i}:")
            traceback.print_exc()
            continue

        states = torch.arange(nt_env.n_states)
        obs = nt_env.get_state(states)
        learned_r = exp_trainer.reward_func(obs.to(torch.float32), None, obs.to(torch.float32), None).detach()
        env_args["R"] = learned_r
        new_env = env_cons(**env_args)

        if policy_name in new_env.policies.keys() and new_env.policies[policy_name] is not None:
            new_policy = new_env.policies[policy_name]
            ret = env.evaluate_policy(new_policy)
            # new_pi = new_env.policies[policy_name].pi
            # pi = env.policies[policy_name].pi
            # kl_div_learned = (-pi * new_pi.log()).sum(dim=-1).mean().item()
            # min_kl_div = (-pi * pi.log()).sum(dim=-1).mean().item()
            score = ret / target_ret if target_ret != 0 else np.nan
            correl = exp_trainer.get_correl(learned_r, env.reward_vector)
            scores.append(score)
            correls.append(correl)

    cfg["scores_mean"] = np.mean(scores)
    cfg["scores_std"] = np.std(scores)
    cfg["correls_mean"] = np.mean(correls)
    cfg["correls_std"] = np.std(correls)

    return cfg
    x = learned_r - learned_r.min()
    ((x / x.max()) - 0.5) * 20
    env.reward_vector


def run_gflow(cfg, nt_env):
    n_epochs = cfg["n_epochs"]
    n_traj = cfg["n_traj"]
    temp = cfg["temp"]
    policy_name = cfg["policy_name"]

    expert_rollouts = nt_env.generate_trajectories(policy_name, n_traj, temp)
    expert_trainer = gflow.GFLOW(
        observation_space=nt_env.observation_space,
        action_space=nt_env.action_space,
        demonstrations=expert_rollouts,
        **filter_params(cfg, gflow.GFLOW)
    )
    expert_trainer.train(n_epochs, log=cfg.get("verbose", False), split_training=cfg.get("split_training", None))
    return expert_trainer


def load_experiment_results():
    df = pd.read_csv("data/experiments.csv")

    df.plot.scatter(x='correls_mean', y='scores_mean')
    plt.show()

    # [k for k,v in ranges.items() if len(v)>1]
    diff_cols = ['target_back_probs', 'reward_type', 'adv_coef', 'n_traj', 'temp', 'val_coef', 'hard', 'use_returns',
                 'use_z']
    corr_cols = ['correls_mean', 'correls_std']
    flt_df = df[df.correls_mean >= .9][diff_cols + corr_cols]
    # significance_matrix(df, 'correls_mean', 'correls_std')

    for i in range(3):
        flt = np.ones(len(flt_df)).astype(bool)
        for col in diff_cols:
            vals, counts = np.unique(flt_df[col], return_counts=True)
            if sum(counts == i) == 1:
                flt = np.bitwise_and(flt, flt_df[col] != vals[counts == i][0])
        flt_df = flt_df[flt]

    df.iloc[90]


def load_experiment_results_2():
    df0 = pd.read_csv("data/experiments.csv")
    df = pd.read_csv("data/experiments_2025-05-06__11-19-36.csv")
    df = pd.read_csv("data/experiments_2025-05-09__02-17-26.csv")
    df = pd.read_csv("data/experiments_2025-05-16__19-36-05.csv")

    for col in df.columns:
        vals, counts = np.unique(df[col], return_counts=True)
        if len(vals) == 1:
            tmp = df0[df0[col] == vals[0]]
            if len(tmp) == 0:
                print("huh?")
            df0 = tmp


def run_individual():
    cfg = dict(
        gamma=0.99,
        net_arch=[64, 64, 64],
        log_prob_loss="kl",
        target_log_probs=True,
        target_back_probs=True,
        reward_type="next state only",
        adv_coef=0.,
        horizon=100,
        n_epochs=300,
        policy_name="Hard Smax",
        n_traj=20,
        temp=1,
        n_trials=3,
        n_states=6,
        lr=1e-3,
        val_coef=0,
        hard=True,
        use_returns=True,
        use_z=True,
        kl_coef=1.,
        use_scheduler=False,
        env_cons=MattGridworld,
        verbose=True,
        split_training=0.3,
        value_is_potential=False,
        # N=2,
        # R=torch.tensor([0.1,0.5,0.3,0.2]),
    )
    cfg = run_config(cfg)
    print(cfg["scores_mean"])
    env_cons = cfg["env_cons"]
    horizon = cfg["horizon"]
    env_args = filter_params(cfg, env_cons)

    env = env_cons(**env_args)
    nt_env = NonTabularMDP(env, horizon)
    exp_trainer = run_gflow(cfg, nt_env)

    exp_trainer


def tmp():
    state = torch.tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]).to(device="cuda", dtype=torch.float32)
    action = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
                          ).to(device="cuda", dtype=torch.float32)
    next_state = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).to(device="cuda", dtype=torch.float32)
    done = torch.zeros_like(action).to(device="cuda", dtype=torch.float32)
    done[-1] = 1.
    next_value = value = None


if __name__ == "__main__":
    # run_individual()
    run_experiment(n_threads=5)
