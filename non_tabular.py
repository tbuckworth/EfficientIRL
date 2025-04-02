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
            s0 = self.get_state(state_idx)
            obs = s0.unsqueeze(0)
            acts = []
            rews = []
            for n in range(self.horizon-1):
                action_probs = pi[state_idx]
                action = self.sample(action_probs)
                next_state_probs = self.T[state_idx,action]
                s = self.get_state(next_state_probs)
                reward = self.R[s]
                obs = torch.cat((obs, s.unsqueeze(0)),dim=0)
                acts.append(action)
                rews.append(reward)
            traj = DictToArgs(dict(
                obs=obs.cpu().numpy(),
                acts=np.array(acts),
                rews=np.array(rews),
            ))
            output.append(traj)
        return output

    def sample(self, x:torch.tensor):
        assert x.sum() >= 1 - 1e-5, "x is not a prob dist"
        r = np.random.rand()
        return (x.cumsum(dim=-1)>r).argwhere()[0]

    def get_state(self, i: int):
        return torch.nn.functional.one_hot(torch.Long(i), self.n_states).to(self.device)

def run():
    gamma = 0.99
    horizon = 10
    n_epochs = 1000
    env = AscenderLong(n_states=6, gamma=gamma)
    run_gflow(env, gamma, horizon, n_epochs)

def run_gflow(env, gamma, horizon, n_epochs):
    nt_env = NonTabularMDP(env, horizon)
    expert_rollouts = nt_env.generate_trajectories("Soft", 10)

    expert_trainer = gflow.GFLOW(
        observation_space=nt_env.observation_space,
        action_space=nt_env.action_space,
        demonstrations=expert_rollouts,
        gamma=gamma,
        net_arch=[8, 8],
    )

    expert_trainer.train(n_epochs)


if __name__ == "__main__":
    run()
