import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# --- Reward Network ---
# Inspired by AIRL’s disentangling: reward is separated from potential shaping.
class RewardNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(RewardNet, self).__init__()
        self.r_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.h_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, s_next=None, gamma=0.99):
        # If s_next is provided, compute the AIRL-form f(s,s')
        r = self.r_net(s)
        h = self.h_net(s)
        if s_next is not None:
            h_next = self.h_net(s_next)
            # f(s, s') = r(s) + gamma * h(s_next) - h(s)
            return r + gamma * h_next - h
        else:
            # For terminal states, we simply use r(s)
            return r #- h


class UpRightPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(UpRightPolicy, self).__init__()

    def forward(self, state):
        return F.softmax(torch.tensor([[100., 100., 0., 0.]]), dim=-1)


# --- Forward Policy ---
class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ForwardPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)


# --- Backward Policy ---
class BackwardPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(BackwardPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)


# --- Simple Grid Environment ---
# A toy grid environment for demonstration. States are 2D (x, y) positions.
class GridEnv:
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.n_actions = 4  # 0: up, 1: right, 2: down, 3: left

    def transition(self, state, action):
        # state: tensor([x, y])
        x, y = int(state[0].item()), int(state[1].item())
        if action == 0:  # up
            y = min(y + 1, self.grid_size[1] - 1)
        elif action == 1:  # right
            x = min(x + 1, self.grid_size[0] - 1)
        elif action == 2:  # down
            y = max(y - 1, 0)
        elif action == 3:  # left
            x = max(x - 1, 0)
        return torch.tensor([float(x), float(y)])


# --- Sampling Trajectories ---
def sample_trajectory(forward_policy, env, init_state, max_steps=10):
    state = init_state
    traj = [state]
    actions = []
    for _ in range(max_steps):
        state_input = state.unsqueeze(0)  # [1, state_dim]
        action_probs = forward_policy(state_input).squeeze(0)  # [action_dim]
        action = torch.multinomial(action_probs, 1).item()
        actions.append(action)
        next_state = env.transition(state, action)
        traj.append(next_state)
        state = next_state
    return traj, actions


# --- Trajectory Balance Loss ---
# The objective is to match the forward-generated trajectory distribution with that defined by the reward.
def trajectory_balance_loss(traj, actions, forward_policy, backward_policy, reward_net, Z, gamma=0.99):
    traj_tensor = torch.stack(traj, dim=0)  # shape: [T+1, state_dim]
    T = traj_tensor.size(0) - 1

    # log_forward = 0.0
    # for t in range(T):
    #     state = traj_tensor[t].unsqueeze(0)  # [1, state_dim]
    #     action = actions[t]
    #     probs = forward_policy(state).squeeze(0)
    #     log_forward += torch.log(probs[action] + 1e-8)

    # same thing, but vectorized:
    probs = forward_policy(traj_tensor)
    log_forwards = torch.log(probs[[i for i in range(len(actions))], actions])

    # For backward, assume a fixed reverse mapping (e.g. up <-> down, right <-> left)
    backward_action_map = {0: 2, 1: 3, 2: 0, 3: 1}
    # log_backward = 0.0
    # for t in range(1, T):
    #     next_state = traj_tensor[t].unsqueeze(0)
    #     # Derive the reverse action from the previous forward action.
    #     fwd_action = actions[t - 1]
    #     back_action = backward_action_map[fwd_action]
    #     probs = backward_policy(next_state).squeeze(0)
    #     log_backward += torch.log(probs[back_action] + 1e-8)

    fwd_action = actions[:-1]
    back_action = [backward_action_map[k] for k in fwd_action]
    probs = backward_policy(traj_tensor[1:-1])
    log_backwards = torch.log(probs[[i for i in range(len(probs))], back_action])
    # # Compute reward from the terminal state.
    # terminal_state = traj_tensor[-1].unsqueeze(0)
    # # For a terminal state we use the simple r(s) output.
    # reward = reward_net(terminal_state)
    # # To ensure positivity, we exponentiate.
    # R_x = torch.exp(reward)

    rewards = reward_net(traj_tensor)
    loss = None
    for i in range(0,traj_tensor.size(0)):
        log_forward = log_forwards[:-i].sum()
        log_backward = log_backwards[:-i].sum()
        reward = rewards[:-i].sum()
        subloss = (log_forward - reward - log_backward) ** 2
        if loss is None:
            loss = subloss
        else:
            loss = loss + subloss
    return loss

    log_forward = log_forwards.sum()
    log_backward = log_backwards.sum()
    reward = rewards.sum()

    # Z is a learnable scalar (normalisation constant)
    log_Z = torch.log(Z + 1e-8)

    # Trajectory Balance: (log(Z) + sum(log forward) - log R(x) - sum(log backward))^2
    loss = (log_Z + log_forward - reward - log_backward) ** 2
    return loss


def main():
    # --- Set-Up ---
    state_dim = 2
    action_dim = 4
    hidden_dim = 64

    reward_net = RewardNet(state_dim, hidden_dim)
    forward_policy = ForwardPolicy(state_dim, action_dim, hidden_dim)
    backward_policy = BackwardPolicy(state_dim, action_dim, hidden_dim)
    expert_policy = UpRightPolicy(state_dim, action_dim)

    # Z as a learnable parameter (initialised to 1.0)
    Z_param = torch.nn.Parameter(torch.tensor(1.0))

    # Optimiser for all parameters
    optimizer = optim.Adam(list(reward_net.parameters()) +
                           list(forward_policy.parameters()) +
                           list(backward_policy.parameters()) +
                           [Z_param], lr=1e-3)

    env = GridEnv(grid_size=(5, 5))
    num_epochs = 50000

    # --- Training Loop ---
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Here we start each trajectory from a fixed point (centre of grid)
        init_state = torch.tensor([0.0, 0.0])
        traj, actions = sample_trajectory(expert_policy, env, init_state, max_steps=15)
        loss = trajectory_balance_loss(traj, actions, forward_policy, backward_policy, reward_net, Z_param)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Z: {Z_param.item():.4f}")
            states = torch.cat([t.unsqueeze(0) for t in traj], dim=0)
            # full_rew = reward_net(states,states[1:])
            rews = reward_net.r_net(states)
            print(torch.cat((states, rews-rews.min()), axis=-1).round(decimals=2))

    print("done")
    full_rews = torch.stack([reward_net(states[i], states[i+1]).detach() for i in range(len(states)-1)],dim=0)

if __name__ == "__main__":
    main()
