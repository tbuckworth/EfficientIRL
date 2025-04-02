from typing import List

import numpy as np
import torch
from einops import einops

import gflow
from helper_local import DictToArgs
from tabular import TabularMDP, AscenderLong
import gymnasium


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

    def generate_trajectories(self, policy_name: str, n_traj: int) -> List[torch.tensor]:
        assert policy_name in self.env.policies.keys(), f"Policy {policy_name} not found in Env {self.env.name}."
        pi = self.env.policies[policy_name].pi
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

    def sample(self, x: torch.tensor):
        assert x.sum() >= 1 - 1e-5, "x is not a prob dist"
        r = np.random.rand()
        return (x.cumsum(dim=-1) > r).argwhere()[0]

    def get_state(self, i):
        if isinstance(i, torch.Tensor):
            t = i
        else:
            t = torch.tensor([i])
        return torch.nn.functional.one_hot(t, self.n_states).to(self.device)


def run():
    gamma = 0.99
    horizon = 10
    n_epochs = 100
    policy_name = "Soft"
    n_traj = 10

    env = AscenderLong(n_states=6, gamma=gamma)
    nt_env = NonTabularMDP(env, horizon)
    exp_trainer = run_gflow(nt_env, gamma, n_epochs, policy_name, n_traj)

    states = torch.arange(nt_env.n_states)
    obs = nt_env.get_state(states)
    learned_r = exp_trainer.reward_func(obs.to(torch.float32), None, obs.to(torch.float32), None).detach()
    new_env = AscenderLong(n_states=env.n_states, gamma=gamma, R=learned_r)

    print(new_env.policies[policy_name].pi.round(decimals=2))
    print(env.policies[policy_name].pi.round(decimals=2))

    return


def run_gflow(nt_env, gamma, n_epochs, policy_name="Soft", n_traj=10):

    expert_rollouts = nt_env.generate_trajectories(policy_name, n_traj)

    expert_trainer = gflow.GFLOW(
        observation_space=nt_env.observation_space,
        action_space=nt_env.action_space,
        demonstrations=expert_rollouts,
        gamma=gamma,
        net_arch=[8, 8],
        log_prob_loss="kl",
        target_log_probs=True,
        reward_type="next state only",
    )

    expert_trainer.train(n_epochs)
    return expert_trainer


if __name__ == "__main__":
    run()
