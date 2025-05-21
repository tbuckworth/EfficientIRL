import re
import time
from abc import ABC

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import scipy.optimize
import einops

from matplotlib import pyplot as plt
import seaborn as sns


def hard_adv_from_belmann(log_pi):
    return log_pi - (log_pi.exp() * log_pi).sum(dim=-1).unsqueeze(-1)


def cosine_similarity_loss(vec1, vec2):
    return 1 - F.cosine_similarity(vec1.reshape(-1), vec2.reshape(-1), dim=-1).mean()


norm_funcs = {
    "l1_norm": lambda x: x if (x == 0).all() else x / x.abs().mean(),
    "l2_norm": lambda x: x if (x == 0).all() else x / x.pow(2).mean().sqrt(),
    "linf_norm": lambda x: x if (x == 0).all() else x / x.abs().max(),
}

dist_funcs = {
    "l1_dist": lambda x, y: (x - y).abs().mean(),
    "l2_dist": lambda x, y: (x - y).pow(2).mean().sqrt(),
}

GAMMA = 0.9


def plot_canonicalised_rewards(canon, hard_canon):
    # 1. Compute the L2 norm of the 'Canonicalised Reward' column for each DataFrame
    canon_l2_norm = np.linalg.norm(canon['Canonicalised Reward'])
    hard_canon_l2_norm = np.linalg.norm(hard_canon['Canonicalised Reward'])

    # 2. Create normalized columns (or Series)
    canon['Normalized Canonicalised Reward'] = canon['Canonicalised Reward'] / canon_l2_norm
    hard_canon['Normalized Canonicalised Reward'] = hard_canon['Canonicalised Reward'] / hard_canon_l2_norm

    # 3. Create a figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Reward
    axes[0].scatter(hard_canon['Reward'], canon['Reward'], alpha=0.6)
    axes[0].set_xlabel('hard_canon: Reward')
    axes[0].set_ylabel('canon: Reward')
    axes[0].set_title('Reward Comparison')

    # Subplot 2: Canonicalised Reward
    axes[1].scatter(hard_canon['Canonicalised Reward'], canon['Canonicalised Reward'], alpha=0.6, color='orange')
    axes[1].set_xlabel('hard_canon: Canonicalised Reward')
    axes[1].set_ylabel('canon: Canonicalised Reward')
    axes[1].set_title('Canonicalised Reward Comparison')

    # Subplot 3: Normalized Canonicalised Reward
    axes[2].scatter(hard_canon['Normalized Canonicalised Reward'],
                    canon['Normalized Canonicalised Reward'],
                    alpha=0.6, color='green')
    axes[2].set_xlabel('hard_canon: Normalized Canonicalised Reward')
    axes[2].set_ylabel('canon: Normalized Canonicalised Reward')
    axes[2].set_title('Normalized Canonicalised Reward Comparison')

    plt.tight_layout()
    plt.show()

def plot_through_time(saved_rewards, true_r, title):
    all_epochs = set()
    for algo_dict in saved_rewards.values():
        all_epochs.update(algo_dict.keys())
    all_epochs = sorted(all_epochs)

    for epoch in all_epochs:
        plt.figure()
        plt.scatter(true_r, true_r, label="True Reward")
        for label, algo_dict in saved_rewards.items():
            if epoch in algo_dict:
                y_coords = np.array(algo_dict[epoch])
            else:
                y_coords = list(algo_dict.values())[np.argwhere(np.array(list(algo_dict.keys())) < epoch).max()]
            correlation = np.corrcoef(true_r, y_coords)[0, 1]  # Compute correlation
            corr_label = f"{label} (corr: {correlation:.2f})"  # Format label with correlation
            plt.scatter(true_r, y_coords, label=corr_label)

        plt.title(f"{title} Epoch {epoch}")
        plt.legend()
        plt.xlabel("True Reward")
        plt.ylabel("Learned Reward")
        plt.show()



class TabularPolicy:
    def __init__(self, name, pi, Q=None, V=None):
        self.name = name
        self.Q = Q
        self.V = V
        self.pi = pi
        # This is in case we have a full zero, we adjust policy.
        flt = (self.pi == 0).any(dim=-1)
        self.pi[flt] = (self.pi[flt] * 10).softmax(dim=-1)
        assert (self.pi.sum(dim=-1).round(
            decimals=3) == 1).all(), "pi is not a probability distribution along final dim"
        self.log_pi = self.pi.log()
        self.R = None
        self.irls = {}


# Define a tabular MDP


class RewardFunc:
    def __init__(self, R, v, next_v, adjustment):
        self.R = R.detach().cpu()
        self.adjustment = adjustment.detach().cpu()
        self.v = v.detach().cpu()
        self.next_v = next_v.detach().cpu()
        self.C = self.R + self.adjustment
        self.n_actions = self.R.shape[-1]
        self.n_states = self.R.shape[-2]
        self.state = torch.arange(self.n_states).unsqueeze(-1).tile(self.n_actions)
        self.action = torch.arange(self.n_actions).unsqueeze(0).tile(self.n_states, 1)
        self.data = {
            "State": self.state.reshape(-1),
            "Action": self.action.reshape(-1),
            "Reward": self.R.reshape(-1),
            "Value": self.v.tile(self.n_actions).reshape(-1),
            "Next Value": self.next_v.reshape(-1),
            "Adjustment": self.adjustment.reshape(-1),
            "Canonicalised Reward": self.C.reshape(-1),
        }
        if len(np.unique([v.shape for k, v in self.data.items()])) > 1:
            raise Exception
        self.df = pd.DataFrame(self.data).round(decimals=2)

    def print(self):
        print(self.df.round(decimals=2))


class TabularMDPs:
    def __init__(self, n_mdps):
        self.pircs = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = n_mdps
        self.n_states = 15
        self.n_actions = 5
        self.T = (torch.rand((self.N, self.n_states, self.n_actions, self.n_states), device=self.device) * 100).softmax(
            dim=-1)
        self.reward_vector = (torch.rand((self.N, self.n_states), device=self.device) * 1000).softmax(dim=-1) * 10
        self.alt_reward_vector = (torch.rand((self.N, self.n_states), device=self.device) * 1000).softmax(dim=-1) * 10

        self.gamma = torch.rand((self.N,), device=self.device) / 10 + 0.9
        belmann_hightemp, belmann_lowtemp = self.q_value_iteration(100000, inv_temp=[1, 1000])
        anti_high, anti_low = self.q_value_iteration(100000, inv_temp=[1, 1000], override_reward=self.alt_reward_vector)
        self.policies = {
            "BelmannHighTemp": belmann_hightemp,
            "BelmannLowTemp": belmann_lowtemp,
            "AlternaHighTemp": anti_high,
            "AlternaLowTemp": anti_low,
        }
        self.norm_funcs = {
            "L2": lambda x: x if (x == 0).all() else x / x.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt(),
            "L1": lambda x: x if (x == 0).all() else x / x.abs().amax(dim=(-1, -2), keepdim=True),
        }
        self.distance = lambda x, y: (x - y).pow(2).mean(dim=(-1, -2)).sqrt()
        self.uniform = torch.ones((self.N, self.n_states, self.n_actions), device=self.device).softmax(dim=-1)

    def q_value_iteration(self, n_iterations=10000, print_message=True, argmax=False, inv_temp=1000,
                          invert_reward=False,
                          override_reward=None):
        T = self.T
        R = self.reward_vector.unsqueeze(-1)
        if invert_reward:
            R *= -1
        if override_reward is not None:
            R = override_reward.unsqueeze(-1)
        gamma = self.gamma.unsqueeze(-1)
        n_states = self.n_states
        n_actions = self.n_actions
        Q = torch.zeros((self.N, n_states, n_actions), device=self.device)

        for i in range(n_iterations):
            old_Q = Q
            V = Q.max(dim=-1).values
            Q = einops.einsum(T, gamma * V, 'N states actions next_states, N next_states -> N states actions') + R

            if (Q - old_Q).abs().max() < 1e-5:
                if print_message:
                    print(f'Q-value iteration converged in {i} iterations')
                if argmax:
                    # NOT SURE THIS WOULD WORK:
                    pi = torch.nn.functional.one_hot(Q.argmax(dim=-2), num_classes=n_actions).float()
                    policy_name = "Hard Argmax"
                else:
                    policy_name = "Hard Smax"
                    if type(inv_temp) == list:
                        # hadv = hard_adv_from_belmann((Q[0] * 1).log_softmax(dim=-1))
                        # ladv = hard_adv_from_belmann((Q[0] * 10).log_softmax(dim=-1))
                        # ratio = ladv/hadv
                        # print(f"mean:{ratio.mean():.2f}\tstd:{ratio.std():.2f}")
                        return [TabularPolicy(policy_name, (Q * t).softmax(dim=-1), Q, V) for t in inv_temp]
                    pi = (Q * inv_temp).softmax(dim=-1)
                return TabularPolicy(policy_name, pi, Q, V)

        print(f"Q-value iteration did not converge in {i} iterations")
        return None

    def value_iteration(self, pi, R, n_iterations: int = 10000):
        T = self.T
        n_states = self.n_states
        V = torch.zeros((self.N, n_states), device=self.device)
        gamma = self.gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        for _ in range(n_iterations):
            old_V = V
            Q = einops.einsum(T, (R + gamma * V.unsqueeze(-2).unsqueeze(-2)), "N s a ns, N s a ns -> N s a")
            V = einops.einsum(pi, Q, "N s a, N s a -> N s")
            if (V - old_V).abs().max() < 1e-5:
                return V
        print(f"Value Iteration did not converge after {n_iterations} iterations.")
        return None


class TabularMDP:
    custom_policies = []

    def __init__(self, n_states, n_actions, transition_prob, reward_vector, mu=None, gamma=GAMMA, name="Unnamed",
                 device=None, custom_only=False):
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.own_pircs = {}
        self.new_pircs = {}
        self.canon = None
        assert (transition_prob.sum(dim=-1) - 1).abs().max() < 1e-5, "Transition Probabilities do not sum to 1"
        self.mu = torch.ones(n_states, device=self.device) / n_states if mu is None else mu.to(device=self.device)
        self.irls = {}
        self.pircs = {}
        self.returns = {}
        self.normalize = norm_funcs["l2_norm"]
        self.distance = dist_funcs["l2_dist"]
        self.name = name
        self.n_states = n_states
        self.n_actions = n_actions
        self.T = transition_prob.to(device=self.device)  # Shape: (n_states, n_actions, n_states)
        self.reward_vector = reward_vector.to(device=self.device)  # Shape: (n_states,)
        self.gamma = gamma

        self.soft_opt = self.soft_q_value_iteration(print_message=False, n_iterations=10000)
        self.hard_opt = self.q_value_iteration(print_message=False, n_iterations=10000, argmax=True)
        self.hard_smax = self.q_value_iteration(print_message=False, n_iterations=100000, argmax=False)

        q_uni = torch.zeros((n_states, n_actions), device=self.device)
        unipi = q_uni.softmax(dim=-1)
        self.uniform = TabularPolicy("Uniform", unipi, q_uni, q_uni.logsumexp(dim=-1))
        # self.hard_adv_cosine = self.hard_adv_learner(print_message=False,
        #                                              n_iterations=10000,
        #                                              criterion=cosine_similarity_loss,
        #                                              name="Hard Adv Cosine")
        # self.hard_adv = self.hard_adv_learner(print_message=False, n_iterations=10000)
        # self.hard_adv_stepped = self.hard_adv_learner_stepped(print_message=False, n_iterations=10000)
        # self.hard_adv_cont = self.hard_adv_learner_continual(print_message=False, n_iterations=10000)
        policies = [self.soft_opt,
                    self.hard_opt,
                    self.hard_smax,
                    self.uniform] + self.custom_policies

        policies = self.custom_policies if custom_only else policies
        self.policies = {p.name: p for p in policies if p is not None}

    def evaluate_policy(self, policy: TabularPolicy):
        R3 = torch.zeros_like(self.T)
        # State reward becomes state, action, next state
        R3[:, :, :] = self.reward_vector
        v = self.value_iteration(policy.pi, R3)

        ret = einops.einsum(self.mu, v, "s, s ->")
        return ret.item()

    def q_value_iteration(self, n_iterations=1000, print_message=True, argmax=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions
        Q = torch.zeros(n_states, n_actions, device=self.device)

        for i in range(n_iterations):
            old_Q = Q
            V = Q.max(dim=1).values
            Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R.unsqueeze(
                1)

            if (Q - old_Q).abs().max() < 1e-8:
                if print_message:
                    print(f'Q-value iteration converged in {i} iterations')
                if argmax:
                    pi = torch.nn.functional.one_hot(Q.argmax(dim=1), num_classes=n_actions).float()
                    policy_name = "Hard Argmax"
                else:
                    policy_name = "Hard Smax"
                    pi = (Q * 1000).softmax(dim=-1)
                return TabularPolicy(policy_name, pi, Q, V)

        print(f"Q-value iteration did not converge in {i} iterations")
        return None

    def hard_adv_learner(self, n_iterations=1000, lr=1e-1, print_message=True, criterion=nn.MSELoss(), name="Hard Adv"):
        T = self.T
        R = self.reward_vector
        n_states = self.n_states
        n_actions = self.n_actions

        logits = torch.rand((n_states, n_actions), requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([logits], lr=lr)
        with torch.no_grad():
            cr = self.canonicalise(R).C.to(device=self.device)

        for i in range(n_iterations):
            old_logits = logits.detach().clone()
            log_pi = logits.log_softmax(dim=-1)
            g = log_pi - (log_pi.exp() * log_pi).sum(dim=-1).unsqueeze(-1)
            loss = criterion(g, cr)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if torch.allclose(logits, old_logits, atol=1e-5):
                if print_message:
                    print(f'hard adv learning converged in {i} iterations')
                pi = logits.softmax(dim=-1).detach()
                return TabularPolicy(name, pi)
        print('hard adv learning did not converge after', n_iterations, 'iterations')
        return None

    def hard_adv_learner_stepped(self, n_iterations=1000, lr=1e-1, print_message=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions

        logits = (torch.ones((n_states, n_actions)) / n_actions).log()
        logits.requires_grad = True

        optimizer = torch.optim.Adam([logits], lr=lr)
        for epoch in range(n_iterations):
            start_logits = logits.detach().clone()
            with torch.no_grad():
                cr = self.canonicalise(R, logits.softmax(dim=-1)).C

            for i in range(n_iterations):
                old_logits = logits.detach().clone()
                log_pi = logits.log_softmax(dim=-1)
                g = log_pi - log_pi.mean(dim=-1).unsqueeze(-1)
                loss = ((g - cr) ** 2).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if torch.allclose(logits, old_logits, atol=1e-5):
                    break
            if torch.allclose(logits, start_logits, atol=1e-5):
                if print_message:
                    print(f'hard adv stepped converged in {epoch} epochs')
                pi = logits.softmax(dim=-1).detach()
                return TabularPolicy("Hard Adv Stepped", pi)
        print('hard adv stepped did not converge after', n_iterations, 'epochs')
        return None

    def hard_adv_learner_continual(self, n_iterations=1000, lr=1e-1, print_message=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions

        logits = (torch.ones((n_states, n_actions)) / n_actions).log()
        logits.requires_grad = True
        optimizer = torch.optim.Adam([logits], lr=lr)
        for i in range(n_iterations):
            start_logits = logits.detach().clone()
            with torch.no_grad():
                cr = self.canonicalise(R, logits.softmax(dim=-1)).C
            log_pi = logits.log_softmax(dim=-1)
            g = log_pi - log_pi.mean(dim=-1).unsqueeze(-1)
            loss = ((g - cr) ** 2).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if torch.allclose(logits, start_logits, atol=1e-5):
                if print_message:
                    print(f'hard adv continual converged in {i} iterations')
                pi = logits.softmax(dim=-1).detach()
                return TabularPolicy("Hard Adv Cont", pi)
        print('hard adv continual did not converge after', n_iterations, 'iterations')
        return None

    def soft_q_value_iteration(self, n_iterations=1000, print_message=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions

        Q = torch.zeros(n_states, n_actions, device=self.device)

        for i in range(n_iterations):
            old_Q = Q
            V = Q.logsumexp(dim=-1)
            Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R.unsqueeze(
                1)

            if (Q - old_Q).abs().max() < 1e-4:
                if print_message:
                    print(f'soft value iteration converged in {i} iterations')
                pi = Q.softmax(dim=1)
                return TabularPolicy("Soft", pi, Q, V)
        print('soft value iteration did not converge after', n_iterations, 'iterations')
        return None

    def calc_pirc(self, policy, pirc_type, own_policy=False):
        trusted_pi = policy.pi if own_policy else None
        if pirc_type in ["Centred", "Centred no C"]:
            adv = policy.log_pi - policy.log_pi.mean(dim=-1).unsqueeze(-1)
            if pirc_type == "Centred":
                policy.centred_canon = self.canonicalise(adv, trusted_pi)
                ca = policy.centred_canon.C
            elif pirc_type == "Centred no C":
                v = torch.zeros((adv.shape[0], 1))
                z = torch.zeros_like(adv)
                policy.centred_no_canon = RewardFunc(adv, v, z, z)
                ca = adv
            else:
                raise NotImplementedError
        elif pirc_type == "Soft":
            adv = policy.log_pi
            policy.soft_canon = self.canonicalise(adv, trusted_pi)
            ca = policy.soft_canon.C
        elif pirc_type == "Hard":
            adv = policy.log_pi - (policy.pi * policy.log_pi).sum(dim=-1).unsqueeze(dim=-1)
            policy.hard_canon = self.canonicalise(adv, trusted_pi)
            ca = policy.hard_canon.C
        else:
            raise NotImplementedError(f"pirc_type must be one of 'Hard','Hard no C','Soft'. Not {pirc_type}.")

        comp_canon = self.canon
        if own_policy:
            comp_canon = self.canonicalise(self.reward_vector, trusted_pi)

        nca = self.normalize(ca)
        ncr = self.normalize(comp_canon.C)

        return self.distance(nca, ncr).item()

    def irl(self, method, time_it=False, verbose=False, atol=1e-5):
        irl_func = irl_funcs[method]
        if verbose:
            print(f"\n{self.name} Environment\t{method}:")
        irls = {}
        true_r = self.reward_vector.cpu().numpy()
        plt.scatter(true_r, true_r, color="black", label="True Reward")
        for name, policy in self.policies.items():
            irl_object = irl_func(
                policy.pi, self.T,
                self.reward_vector,
                self.mu, device=self.device,
                suppress=True, atol=atol,
                n_iterations=20000,
            )
            reward, elapsed = irl_object.train()
            irl_object.display_rewards()
            policy.irls[method] = irl_object
            if verbose:
                print(f"{name}\tIRL: {reward:.4f}\tElapsed: {elapsed:.4f}")
            irls[name] = {"IRL": reward.item(), "Time": elapsed} if time_it else reward.item()
        plt.legend()
        plt.title(f"{self.name}")
        plt.show()
        self.irls[method] = irls

    def calc_irls_by_policy(self, plot_final=True, time_it=False, atol=1e-5):
        for p_name, policy in self.policies.items():
            true_r = self.reward_vector.cpu().numpy()
            if plot_final:
                plt.scatter(true_r, true_r, color="black", label="True Reward")
            else:
                saved_rewards = {}

            for irl_name, irl_func in irl_funcs.items():
                irl_object = irl_func(
                    policy.pi,
                    self.T,
                    self.reward_vector,
                    self.mu,
                    device=self.device,
                    suppress=True,
                    atol=atol,
                    n_iterations=10000,
                )
                reward, elapsed = irl_object.train()
                if plot_final:
                    irl_object.display_rewards()
                else:
                    saved_rewards[irl_name] = irl_object.saved_rewards
            if plot_final:
                plt.legend()
                plt.title(f"Learned rewards for {p_name} Policy in {self.name}")
                plt.savefig(f"data/tabular_plots/{self.name}_{p_name}_policy.png")
                plt.show()
            else:
                plot_through_time(saved_rewards, true_r, f"Learned rewards for {p_name} Policy in {self.name}")



    def calc_irls(self, verbose=False, time_it=False, atol=1e-5):
        for irlf in irl_funcs.keys():
            self.irl(irlf, time_it=time_it, atol=atol)
        if time_it:
            df = pd.concat(
                {person: pd.DataFrame(methods).T for person, methods in self.irls.items()},
                axis=1
            ).round(decimals=2)
        else:
            df = pd.DataFrame(self.irls).round(decimals=2)
        if verbose:
            print(f"{self.name} Environment")
            print(df)
        return df

    def calc_returns(self, verbose=False):
        returns = {}
        for name, policy in self.policies.items():
            returns[name] = self.evaluate_policy(policy)
        self.returns["Centred"] = returns
        df = pd.DataFrame(self.returns).round(decimals=2)
        if verbose:
            print(f"{self.name} Environment")
            print(df)
        return df

    def canonicalise(self, R, trusted_pi=None):
        if trusted_pi is None:
            trusted_pi = self.uniform.pi

        R3 = torch.zeros_like(self.T)

        if R.ndim == 1:
            # State reward becomes state, action, next state
            R3[:, :, :] = R
        elif R.ndim == 2:
            # State action reward becomes state, action, next state
            R3 = R.unsqueeze(-1).tile(self.T.shape[-1])
        elif R.ndim == 3:
            R3 = R
        else:
            raise Exception(f"R.ndim must be 1, 2, or 3, not {R.ndim}.")

        v = self.value_iteration(trusted_pi, R3)

        R2 = (R3 * self.T).sum(dim=-1)

        next_v = (self.T * v.view(1, 1, -1)).sum(dim=-1)
        adjustment = self.gamma * next_v - v.view(-1, 1)

        return RewardFunc(R2, v.view(-1, 1), next_v, adjustment)

    def value_iteration(self, pi, R, n_iterations: int = 10000):
        T = self.T
        n_states = self.n_states
        V = torch.zeros((n_states), device=self.device)

        for _ in range(n_iterations):
            old_V = V
            Q = einops.einsum(T, (R + self.gamma * V.view(1, 1, -1)), "s a ns, s a ns -> s a")
            V = einops.einsum(pi, Q, "s a, s a -> s")
            if (V - old_V).abs().max() < 1e-5:
                return V
        print(f"Value Iteration did not converge after {n_iterations} iterations.")
        return None


class IRLFunc(ABC):
    param_list = []
    no_optimizer = False
    name = "Unnamed"
    convergence_type = None
    use_scheduler = None

    def __init__(self, pi, T, true_reward, mu=None, n_iterations: int = 10000, lr=1e-2, print_losses=False,
                 device="cpu",
                 suppress=False, atol=1e-5,
                 state_based=True, soft=True, use_scheduler=False):
        self.n_states, self.n_actions, _ = T.shape
        self.n_iterations = n_iterations
        self.print_losses = print_losses
        self.device = device
        self.suppress = suppress
        self.atol = atol
        self.state_based = state_based
        self.soft = soft
        self.true_reward = true_reward
        self.mu = mu
        self.T = T
        self.pi = pi.to(device)
        self.log_pi = self.pi.log()
        self.log_pi.requires_grad = False
        self.T.requires_grad = False
        self.use_scheduler = use_scheduler
        self.optimizer = torch.optim.Adam(self.param_list, lr=lr)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_iterations,
                                                                        eta_min=0)
        self.max_ent = np.log(1 / self.n_actions)
        self.converged = self.meg = self.log_pi_soft_less_max_ent = None
        self.saved_rewards = {}

    def train(self):
        start = time.time()
        for i in range(self.n_iterations):

            old_params = [p.detach().clone() for p in self.param_list]

            loss = self.calculate_loss()
            loss.backward()
            self.optimizer.step()
            if self.use_scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            if self.print_losses and i % 10 == 0 and loss is not None:
                print(f"Loss:{loss.item():.4f}")
            if i % 1000 == 0 or i == self.n_iterations - 1:
                self.saved_rewards[i] = self.get_state_based_learned_reward().detach().cpu().numpy()
            if self.check_convergence(old_params):
                if not self.suppress:
                    print(f'{self.name} IRL converged in {i} iterations.')
                self.converged = True
                self.saved_rewards[i] = self.get_state_based_learned_reward().detach().cpu().numpy()
                return self.learned_reward, time.time() - start
        print(f'{self.name} IRL did not converge in {i} iterations')
        self.converged = False
        return self.learned_reward, time.time() - start

    def check_convergence(self, old_params):
        return np.all([torch.allclose(p, old_p, atol=self.atol) for p, old_p in zip(self.param_list, old_params)])

    def calculate_loss(self):
        raise NotImplementedError("calculate_loss is an abstract method. It must be overidden.")

    def get_state_based_learned_reward(self):
        raise NotImplementedError("get_state_based_learned_reward is an abstract method. It must be overidden.")

    def display_rewards(self):
        learned_reward = self.get_state_based_learned_reward()
        corr = torch.corrcoef(torch.stack((self.true_reward, learned_reward)))[0, 1]
        plt.scatter(x=self.true_reward.cpu().numpy(),
                    y=learned_reward.detach().cpu().numpy(),
                    label=f"{self.name} Corr:{corr:.2f}")
        # plt.legend()
        # plt.title(f"{self.name}")
        # plt.show()
        print(f"{self.name}")
        print(learned_reward)
        print(self.true_reward)


class AlternativeEIRL(IRLFunc):
    def __init__(self, pi, T, true_reward, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, device="cpu",
                 suppress=False,
                 atol=1e-5,
                 state_based=False, soft=True):
        self.consistency_coef = 10.
        n_states, n_actions, _ = T.shape
        self.learned_log_pi = torch.randn((n_states, n_actions), requires_grad=True, device=device)
        self.ns_log_pi = torch.randn((n_states,), requires_grad=True, device=device)
        self.learned_value = torch.randn((n_states,), requires_grad=True, device=device)
        self.learned_reward = None
        self.param_list = [self.learned_log_pi, self.ns_log_pi, self.learned_value]
        self.name = "Alternative EIRL"
        super().__init__(pi, T, true_reward, mu, n_iterations, lr, print_losses, device, suppress, atol, state_based,
                         soft)

    def calculate_loss(self):
        log_pi_theta = self.learned_log_pi.log_softmax(dim=-1)

        next_value = einops.einsum(self.T, self.learned_value, "s a ns, ns -> s a")
        log_pi_target = einops.einsum(self.T, self.ns_log_pi, "s a ns, ns -> s a")

        self.learned_reward = log_pi_target - GAMMA * next_value + self.learned_value.unsqueeze(-1)

        loss1 = -(self.pi * log_pi_theta).mean()
        loss2 = (log_pi_theta - log_pi_target).pow(2).mean()
        loss3 = log_pi_target.pow(2).mean()
        loss = loss1 + loss2 * self.consistency_coef + loss3
        return loss

    def get_state_based_learned_reward(self):
        return einops.einsum(self.T, self.learned_reward, "s a ns, s a ->ns")


class NextStateEIRL(IRLFunc):
    def __init__(self, pi, T, true_reward, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, device="cpu",
                 suppress=False,
                 atol=1e-5,
                 state_based=False, soft=True):
        self.consistency_coef = 10.
        n_states, n_actions, _ = T.shape
        self.learned_log_pi = torch.randn((n_states, n_actions), requires_grad=True, device=device)
        self.learned_reward = torch.randn((n_states, 1), requires_grad=True, device=device)
        self.learned_value = torch.randn((n_states,), requires_grad=True, device=device)

        self.param_list = [self.learned_log_pi, self.learned_reward, self.learned_value]
        self.name = "Next State EIRL"
        super().__init__(pi, T, true_reward, mu, n_iterations, lr, print_losses, device, suppress, atol, state_based,
                         soft)

    def calculate_loss(self):
        log_pi_theta = self.learned_log_pi.log_softmax(dim=-1)
        actor_adv = log_pi_theta
        if not self.soft:
            # Adding entropy makes it hard advantage
            actor_adv = log_pi_theta - (log_pi_theta.exp() * log_pi_theta).sum(dim=-1).unsqueeze(dim=-1)

        next_value = einops.einsum(self.T, self.learned_value, "s a ns, ns -> s a")
        reward_hat = einops.einsum(self.T, self.learned_reward.squeeze(), "s a ns, ns -> s a")
        q_hat = reward_hat + GAMMA * next_value

        reward_adv = q_hat - self.learned_value.unsqueeze(-1)

        loss1 = -(self.pi * log_pi_theta).mean()
        loss2 = (actor_adv - reward_adv).pow(2).mean()
        loss = loss1 + loss2 * self.consistency_coef
        return loss

    def get_state_based_learned_reward(self):
        return self.learned_reward.squeeze()  # einops.einsum(self.T, self.learned_reward, "s a ns, ns ->ns")


class DisentangledEIRL(IRLFunc):
    def __init__(self, pi, T, true_reward, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, device="cpu",
                 suppress=False,
                 atol=1e-5,
                 state_based=False, soft=True):
        self.consistency_coef = 10.
        n_states, n_actions, _ = T.shape
        self.learned_log_pi = torch.randn((n_states, n_actions), requires_grad=True, device=device)
        reward_shape = (n_states, n_actions)
        if state_based:
            reward_shape = (n_states, 1)
        self.learned_reward = torch.randn(reward_shape, requires_grad=True, device=device)
        self.learned_value = torch.randn((n_states,), requires_grad=True, device=device)

        self.param_list = [self.learned_log_pi, self.learned_reward, self.learned_value]
        self.name = "Disentangled EIRL"
        super().__init__(pi, T, true_reward, mu, n_iterations, lr, print_losses, device, suppress, atol, state_based,
                         soft)

    def calculate_loss(self):
        log_pi_theta = self.learned_log_pi.log_softmax(dim=-1)
        actor_adv = log_pi_theta
        if not self.soft:
            # Adding entropy makes it hard advantage
            actor_adv = log_pi_theta - (log_pi_theta.exp() * log_pi_theta).sum(dim=-1).unsqueeze(dim=-1)

        next_value = einops.einsum(self.T, self.learned_value, "s a ns, ns -> s a")

        q_hat = self.learned_reward + GAMMA * next_value

        reward_adv = q_hat - self.learned_value.unsqueeze(-1)

        loss1 = -(self.pi * log_pi_theta).mean()
        loss2 = (actor_adv - reward_adv).pow(2).mean()
        loss = loss1 + loss2 * self.consistency_coef
        return loss

    def get_state_based_learned_reward(self):
        return einops.einsum(self.T, self.learned_reward, "s a ns, s a ->ns")


class StateBasedDisentangledEIRL(DisentangledEIRL):
    def __init__(self, *args, **kwargs):
        kwargs["state_based"] = True
        super().__init__(*args, **kwargs)
        self.name = "State Based Disentangled EIRL"

    def get_state_based_learned_reward(self):
        return einops.einsum(self.T, self.learned_reward.squeeze(), "s a ns, s ->ns")


class HardDisentangledEIRL(DisentangledEIRL):
    def __init__(self, *args, **kwargs):
        kwargs["soft"] = False
        super().__init__(*args, **kwargs)
        self.name = "Hard Disentangled EIRL"


class HardStateBasedDisentangledEIRL(StateBasedDisentangledEIRL):
    def __init__(self, *args, **kwargs):
        kwargs["soft"] = False
        super().__init__(*args, **kwargs)
        self.name = "Hard State Based Disentangled EIRL"


class NextValEIRL(IRLFunc):
    def __init__(self, pi, T, true_reward, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, device="cpu",
                 suppress=False,
                 atol=1e-5,
                 state_based=True, soft=True):
        self.consistency_coef = 10.
        n_states, n_actions, _ = T.shape
        self.learned_log_pi = torch.randn((n_states, n_actions), requires_grad=True, device=device)
        self.learned_reward = torch.randn((n_states,), requires_grad=True, device=device)

        self.param_list = [self.learned_log_pi, self.learned_reward]
        self.name = "Next Value EIRL"
        super().__init__(pi, T, true_reward, mu, n_iterations, lr, print_losses, device, suppress, atol, state_based,
                         soft)

    def calculate_loss(self):
        log_pi_theta = self.learned_log_pi.log_softmax(dim=-1)
        target = einops.einsum(self.T, self.learned_reward, "s a ns, ns -> s a")
        loss1 = -(self.pi * log_pi_theta).mean()
        loss2 = (log_pi_theta - target).pow(2).mean()
        loss = loss1 + loss2 * self.consistency_coef
        return loss


# Define an epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon):
    """Generates an epsilon-greedy policy given Q-values."""
    n_states, n_actions = q_values.shape
    greedy_policy = torch.zeros((n_states, n_actions))
    greedy_actions = q_values.argmax(dim=1)
    greedy_policy[torch.arange(n_states), greedy_actions] = 1 - epsilon
    greedy_policy += epsilon / n_actions
    return greedy_policy


class AscenderLong(TabularMDP):
    def __init__(self, n_states, stochastic=False, gamma=GAMMA, R=None):
        assert n_states % 2 == 0, (
            "Ascender requires a central starting state with an equal number of states to the left"
            " and right, plus an infinite terminal state. Therefore n_states must be even.")
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        p = 1 if not stochastic else 0.9
        for i in range(n_states - 3):
            T[i + 1, 1, i + 2] = p
            T[i + 1, 0, i] = p

            T[i + 1, 1, i] = 1 - p
            T[i + 1, 0, i + 2] = 1 - p

        T[(0, -1, -2), :, -1] = 1  # /n_actions

        if R is None:
            R = torch.zeros(n_states)
            R[-2] = 10
            R[0] = -10
        assert(len(R) == n_states)

        mu = torch.zeros(n_states)
        mu[(n_states - 1) // 2] = 1.

        go_left = torch.zeros((n_states, n_actions))
        go_left[:, 0] = 1
        go_left[-1] = 0.5
        self.go_left = TabularPolicy("Go Left", go_left)
        self.custom_policies = [self.go_left]
        self.horizon = n_states * 2
        super().__init__(n_states, n_actions, T, R, mu, gamma, f"Ascender: {int((n_states - 2) // 2)} Pos States")


class OneStep(TabularMDP):
    def __init__(self, gamma=GAMMA, R=None):
        n_states = 5
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        T[:2, 0, 2] = 1
        T[:2, 1, 3] = 1
        T[2:, :, -1] = 1  # /n_actions
        if R is None:
            R = torch.zeros(n_states)
            R[2] = 1
        mu = torch.zeros(n_states)
        mu[:2] = 0.5

        consistent_pi = torch.zeros(n_states, n_actions)
        consistent_pi[:2] = torch.FloatTensor([0.2, 0.8])
        consistent_pi[2:] = 0.5

        inconsistent_pi = consistent_pi.clone()
        inconsistent_pi[1] = torch.FloatTensor([0.8, 0.2])
        self.consistent = TabularPolicy("Consistent", consistent_pi)
        self.inconsistent = TabularPolicy("Inconsistent", inconsistent_pi)
        self.custom_policies = [self.consistent, self.inconsistent]
        self.horizon = 2
        super().__init__(n_states, n_actions, T, R, mu, gamma, "One Step")


class DiffParents(TabularMDP):
    def __init__(self, gamma=GAMMA, custom_only=False, R=None):
        n_states = 6
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        T[0, 0, 2] = 1
        T[0, 1, 1] = 1
        T[(1, 2, 4, 5), :, -1] = 1
        T[3, 0, 1] = 1
        T[3, 1, 4] = 1

        if R is None:
            R = torch.zeros(n_states)
            R[1] = 1
        mu = torch.zeros(n_states)
        mu[(0, 3),] = 0.5

        consistent_pi = torch.zeros(n_states, n_actions)
        consistent_pi[0] = torch.FloatTensor([0.2, 0.8])
        consistent_pi[3] = torch.FloatTensor([0.7, 0.3])

        consistent_pi[(1, 2, 4, 5),] = 0.5

        inconsistent_pi = consistent_pi.clone()
        inconsistent_pi[0] = torch.FloatTensor([0.8, 0.2])
        self.consistent = TabularPolicy("Consistent", consistent_pi)
        self.inconsistent = TabularPolicy("Inconsistent", inconsistent_pi)
        self.custom_policies = [self.consistent, self.inconsistent]
        super().__init__(n_states, n_actions, T, R, mu, gamma, "DiffParents", custom_only=custom_only)


class OneStepOther(TabularMDP):
    def __init__(self, gamma=GAMMA):
        n_states = 7
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        T[:3, 0, 3] = 1
        T[:2, 1, 4] = 1
        T[2, 1, 5] = 1
        T[3:, :, -1] = 1  # /n_actions

        R = torch.zeros(n_states)
        R[3] = 1
        R[5] = 10
        mu = torch.zeros(n_states)
        mu[:3] = 1 / 3

        consistent_pi = torch.zeros(n_states, n_actions)
        consistent_pi[:3] = torch.FloatTensor([0.8, 0.2])
        consistent_pi[3:] = 0.5

        inconsistent_pi = consistent_pi.clone()
        inconsistent_pi[2] = torch.FloatTensor([0.1, 0.9])
        self.consistent = TabularPolicy("Consistent", consistent_pi)
        self.inconsistent = TabularPolicy("Inconsistent", inconsistent_pi)
        self.custom_policies = [self.consistent, self.inconsistent]
        super().__init__(n_states, n_actions, T, R, mu, gamma, "One Step Other")


class DogSatMat(TabularMDP):
    def __init__(self, gamma=GAMMA, R=None):
        # The _ sat on the _
        # actions are dog/cat/mat/floor.
        chars = "_DCMF"
        states = {}
        actions = {}
        counter = 0
        act_counter = -1
        for c0 in chars:
            actions[f"_{c0}"] = act_counter
            act_counter += 1
            actions[f"{c0}_"] = act_counter
            act_counter += 1
            for c2 in chars:
                states[c0 + c2] = counter
                counter += 1
        n_states = len(states)
        n_actions = len(actions)
        T = torch.zeros(n_states, n_actions, n_states)

        for state, s_idx in states.items():
            for action, a_idx in actions.items():
                next_state = [c for c in state]
                if next_state[0] == '_':
                    next_state[0] = action[0]
                if next_state[1] == '_':
                    next_state[1] = action[1]
                ns_idx = states[''.join(next_state)]
                T[s_idx, a_idx, ns_idx] = 1

        if R is None:
            R = torch.zeros(n_states)
            for state in ["DM", "CM"]:
                R[states[state]] = np.log(9)
            for state in ["DF", "CF"]:
                R[states[state]] = np.log(1)

        mu = torch.zeros(n_states)
        mu[states["__"]] = 1

        llm_pi = torch.zeros(n_states, n_actions)
        llm_pi[states["_M"], actions["D_"]] = 0.5
        llm_pi[states["_M"], actions["C_"]] = 0.5
        llm_pi[states["_F"], actions["D_"]] = 0.5
        llm_pi[states["_F"], actions["C_"]] = 0.5
        llm_pi[states["D_"], actions["_M"]] = 0.9
        llm_pi[states["D_"], actions["_F"]] = 0.1
        llm_pi[states["C_"], actions["_M"]] = 0.9
        llm_pi[states["C_"], actions["_F"]] = 0.1
        flt = llm_pi.sum(dim=-1) == 0
        llm_pi[flt] = 1 / n_actions
        self.llm = TabularPolicy("LLM", llm_pi)
        self.custom_policies = [self.llm]
        self.horizon = 4
        super().__init__(n_states, n_actions, T, R, mu, gamma, "Dog Sat Mat")


class RandMDP(TabularMDP):
    def __init__(self, gamma=GAMMA):
        n_states = np.random.randint(2, 10)
        n_actions = np.random.randint(2, 4)
        T = torch.randn(n_states, n_actions, n_states).softmax(dim=-1)
        R = (torch.rand(n_states) * np.random.randint(1, 100)).softmax(dim=0) * 10
        mu = (torch.rand(n_states) * np.random.randint(1, 100)).softmax(dim=0)
        super().__init__(n_states, n_actions, T, R, mu, gamma, "Rand")


class CustMDP(TabularMDP):
    def __init__(self, gamma=GAMMA, R=None):
        n_states = 6
        n_actions = 2
        T = torch.tensor(
            [[[0.4488, 0.0894, 0.0317, 0.1196, 0.0744, 0.2361],
              [0.0721, 0.2445, 0.1476, 0.3135, 0.0818, 0.1404]],
             [[0.2993, 0.1653, 0.1319, 0.3343, 0.0444, 0.0247],
              [0.2658, 0.0641, 0.0275, 0.0427, 0.2545, 0.3455]],
             [[0.1973, 0.0531, 0.1714, 0.3228, 0.0961, 0.1593],
              [0.4082, 0.0361, 0.0068, 0.2510, 0.2770, 0.0209]],
             [[0.2416, 0.1333, 0.0108, 0.2360, 0.2142, 0.1641],
              [0.1593, 0.3816, 0.0565, 0.0877, 0.1874, 0.1275]],
             [[0.0927, 0.3711, 0.1062, 0.1825, 0.0417, 0.2057],
              [0.1423, 0.0786, 0.0535, 0.4766, 0.1001, 0.1488]],
             [[0.0330, 0.0479, 0.0361, 0.3026, 0.5278, 0.0526],
              [0.4085, 0.1154, 0.2842, 0.0448, 0.0637, 0.0835]]])
        if R is None:
            R = torch.tensor([1.3675e-05, 1.1792e-03, 1.2642e-06, 1.0816e-01, 2.5503e-05, 9.8906e+00])
        mu = torch.tensor([0.0686, 0.1983, 0.1976, 0.1412, 0.1533, 0.2409]).log().softmax(dim=0)
        T = T.log().softmax(dim=-1)
        self.horizon = 50
        super().__init__(n_states, n_actions, T, R, mu, gamma, "Weird Failure")


class MattGridworld(TabularMDP):
    def __init__(self, gamma=GAMMA, N=5, R=None):
        n_states = N * N

        actions = ['up', 'down', 'left', 'right']
        n_actions = len(actions)

        T = torch.zeros((n_states, n_actions, n_states))

        def state_index(x, y):
            return x * N + y

        def state_coords(s):
            return divmod(s, N)

        for x in range(N):
            for y in range(N):
                s = state_index(x, y)
                for a_idx, action in enumerate(actions):
                    if action == 'up':
                        nx, ny = x - 1, y
                    elif action == 'down':
                        nx, ny = x + 1, y
                    elif action == 'left':
                        nx, ny = x, y - 1
                    elif action == 'right':
                        nx, ny = x, y + 1

                    if action == 'up':
                        ox, oy = x + 1, y
                    elif action == 'down':
                        ox, oy = x - 1, y
                    elif action == 'left':
                        ox, oy = x, y + 1
                    elif action == 'right':
                        ox, oy = x, y - 1

                    if 0 <= nx < N and 0 <= ny < N:
                        ns_intended = state_index(nx, ny)
                    else:
                        ns_intended = s

                    if 0 <= ox < N and 0 <= oy < N:
                        ns_opposite = state_index(ox, oy)
                    else:
                        ns_opposite = s

                    T[s, a_idx, ns_intended] += 0.9
                    T[s, a_idx, ns_opposite] += 0.1
        if R is None:
            R = torch.rand(n_states)
        mu = torch.zeros(n_states)
        mu[0] = 1
        self.horizon = n_states * 4
        super().__init__(n_states, n_actions, T, R, mu, gamma, "Matt Gridworld")


irl_func_list = [DisentangledEIRL, NextValEIRL]

irl_funcs = {
    # "Alternative EIRL": AlternativeEIRL,
    "Next State Disentangled EIRL": NextStateEIRL,
    "Disentangled EIRL": DisentangledEIRL,
    # "State Based Disentangled EIRL": StateBasedDisentangledEIRL,
    # "Hard Disentangled EIRL": HardDisentangledEIRL,
    # "Hard State Based Disentangled EIRL": HardStateBasedDisentangledEIRL,
    # "Next Value EIRL": NextValEIRL,
}


def cust_mpd():
    while True:
        CustMDP().calc_irls(verbose=True, time_it=False, atol=1e-4)


def random_mdp():
    for i in range(100):
        RandMDP().calc_irls(verbose=True, time_it=False, atol=1e-4)


def timing():
    csv_file = "data/meg_timings_stochastic.csv"

    def get_stats(IRLConstructor, policy, env, atol, convergence_type, use_scheduler, seed):
        learner = IRLConstructor(policy.pi, env.T, env.mu, device=env.device, suppress=True, atol=atol,
                                 n_iterations=100000,
                                 convergence_type=convergence_type,
                                 use_scheduler=use_scheduler)
        meg, elapsed = learner.train()
        converged = learner.converged
        return [{"Type": learner.name,
                 "IRL": meg.item(),
                 "Elapsed": elapsed,
                 "Convergence_Type": convergence_type,
                 "Scheduler": use_scheduler,
                 "Converged": converged,
                 "atol": atol,
                 "seed": seed,
                 "n_states": env.n_states}]

    outputs = []
    policy_name = "Hard Smax"
    for atol in [1e-3, 1e-5]:
        for i in [10, 20, 50, 76, 100, 150]:
            for convergence_type in ["Q"]:
                for seed in [42, 6033, 0, 100, 500]:
                    np.random.seed(seed)
                    env = AscenderLong(n_states=i, stochastic=True)
                    policy = env.policies[policy_name]
                    outputs += get_stats(KLDivIRL, policy, env, atol, convergence_type, use_scheduler=True, seed=seed)
                    # outputs += get_stats(KLDivIRL, policy, env, atol, convergence_type, use_scheduler=False)
                    outputs += get_stats(MattIRL, policy, env, atol, convergence_type, use_scheduler=None, seed=seed)
            df = pd.DataFrame(outputs)
            print(df)
    df = pd.DataFrame(outputs)
    df.to_csv(csv_file, index=False)
    print(df)
    plot_timings(csv_file)


def plot_timings(csv_dir):
    save_dir = re.sub(r"\.csv", ".png", csv_dir)
    # Load the CSV file
    df = pd.read_csv(csv_dir)

    # Remove outliers where n_states = 200
    df_filtered = df[df["n_states"] != 150]

    # Define an exponential function for curve fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Get unique atol values
    unique_atol = df_filtered["atol"].unique()

    # Create subplots for different atol values with exponential fits
    fig, axes = plt.subplots(2, len(unique_atol), figsize=(12, 12), sharey='row')

    # Plot for each atol with exponential curve fits (Elapsed Time)
    for ax, atol_value in zip(axes[0], unique_atol):
        subset = df_filtered[df_filtered["atol"] == atol_value]

        # Scatter plot
        sns.scatterplot(
            data=subset,
            x="n_states",
            y="Elapsed",
            hue="Type",
            palette="viridis",
            ax=ax
        )

        # Fit and plot exponential curves
        for t in subset["Type"].unique():
            type_subset = subset[subset["Type"] == t]
            x_data = type_subset["n_states"].values
            y_data = type_subset["Elapsed"].values

            # Fit the exponential function
            try:
                popt, _ = scipy.optimize.curve_fit(exp_func, x_data, y_data, p0=(1, 0.01, 1))
                x_fit = np.linspace(min(x_data), max(x_data), 100)
                y_fit = exp_func(x_fit, *popt)
                ax.plot(x_fit, y_fit, label=f"{t} Fit")
            except:
                pass  # Skip if fitting fails

        ax.set_title(f"atol = {atol_value}")
        ax.set_xlabel("Number of States")

    # Plot for each atol (IRL values)
    for ax, atol_value in zip(axes[1], unique_atol):
        subset = df_filtered[df_filtered["atol"] == atol_value]

        # Scatter plot
        sns.scatterplot(
            data=subset,
            x="n_states",
            y="IRL",
            hue="Type",
            palette="viridis",
            ax=ax
        )

        ax.set_title(f"atol = {atol_value} (IRL)")
        ax.set_xlabel("Number of States")

    # Set common y-axis labels
    axes[0, 0].set_ylabel("Elapsed Time")
    axes[1, 0].set_ylabel("IRL Value")

    # Show the plot
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()


def main():
    envs = [
        MattGridworld(),
        AscenderLong(n_states=6),
        DogSatMat(),
        CustMDP(),
        # OneStepOther(),
        OneStep(),
        # DiffParents(),

    ]
    # envs = [MattGridworld()]
    envs = {e.name: e for e in envs}

    for name, env in envs.items():
        df = env.calc_irls_by_policy(plot_final=False, time_it=False, atol=1e-8)
        # print(df)
        # print(f"\n{name}:\n{env.meg_pirc()}")

    return


if __name__ == "__main__":
    # timing()
    # cust_mpd()
    # random_mdp()
    main()
    # gridworld_analysis()
    # vMDP()
    # try_hard_adv_train()
