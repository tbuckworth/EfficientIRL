import numpy as np
import torch
from imitation.algorithms import base
from imitation.rewards.reward_nets import BasicShapedRewardNet, BasicRewardNet, ShapedRewardNet
from jinja2 import optimizer
from stable_baselines3.common.policies import ActorCriticPolicy

from helper_local import get_policy_for


class CustomShapedRewardNet(ShapedRewardNet):
    def forward(self, state, action, next_state, done, value=None, next_value=None):
        base_reward_net_output = self.base(state, action, next_state, done)
        if next_value is None:
            next_value = self.potential(next_state).flatten()
        if value is None:
            value = self.potential(state).flatten()

        new_shaping = (1 - done.float()) * next_value.squeeze()
        final_rew = (
                base_reward_net_output
                + self.discount_factor * new_shaping
                - value.squeeze()
        )
        assert final_rew.shape == state.shape[:1]
        return final_rew.squeeze(), base_reward_net_output.squeeze()

    def forward_trajectory(self, state, action, done, value=None):
        next_value = None
        if value is not None:
            value = value.squeeze()
            shp = [i for i in value.shape]
            shp[-1] = 1
            zeros = torch.zeros(shp).to(device=self.device)
            next_value = torch.cat((value[..., 1:], zeros), dim=-1)
        return self.forward(state[:-1], action, state[1:], done, value, next_value)


class GFLOW:
    def __init__(self,
                 observation_space,
                 action_space,
                 demonstrations,
                 gamma=0.98,
                 batch_size=256,
                 lr=1e-3,
                 l2_weight=0.001,
                 rng=None,
                 custom_logger=None,
                 net_arch=None,
                 use_state=True,
                 use_action=False,
                 use_next_state=False,
                 device=None,
                 val_coef=1.):
        self.val_coef = val_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net_arch is None:
            net_arch = [256, 256, 256, 256]
        self.observation_space = observation_space
        self.action_space = action_space
        self.demonstrations = demonstrations
        self.gamma = gamma
        self.batch_size = batch_size
        self.l2_weight = l2_weight
        self.rng = rng
        self.custom_logger = custom_logger

        self.Z_param = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_policy = get_policy_for(observation_space, action_space, net_arch)
        self.backward_policy = get_policy_for(observation_space, action_space, net_arch)

        value_func = lambda obs: self.forward_policy.predict_values(obs).squeeze()

        self.output_reward_function = BasicRewardNet(
            observation_space,
            action_space,
            use_state,
            use_action,
            use_next_state
        )
        self.reward_net = CustomShapedRewardNet(
            self.output_reward_function,
            value_func,
            gamma)
        nets = [self.forward_policy, self.backward_policy, self.output_reward_function, self.Z_param]
        _ = [n.to(device=self.device) for n in nets]
        self.optimizer = torch.optim.Adam(list(self.forward_policy.parameters()) +
                                          list(self.backward_policy.parameters()) +
                                          list(self.output_reward_function.parameters()) +
                                          list(self.reward_net.parameters()) +
                                          [self.Z_param],
                                          lr=lr)
        # self.data = torch.utils.data.DataLoader(
        #     demonstrations,
        #     batch_size=batch_size,
        #     # collate_fn=types.transitions_collate_fn,
        #
        # )

    @property
    def policy(self) -> ActorCriticPolicy:
        return self.forward_policy

    @property
    def reward_func(self) -> BasicRewardNet:
        return self.output_reward_function

    def train(self, n_epochs, progress_bar=None):
        for epoch in range(n_epochs):
            self.custom_logger.record("epoch", epoch)
            # maybe shuffle?
            for traj in self.demonstrations:
                loss, stats = self.trajectory_balance_loss(traj)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.log(stats)

    def trajectory_balance_loss(self, traj):
        obs = torch.FloatTensor(traj.obs).to(device=self.device)
        acts = torch.FloatTensor(traj.acts).to(device=self.device)
        infos = traj.infos
        done = torch.zeros_like(acts).to(device=self.device)
        # if we batch trajs, then will have to modify this
        done[-1] = 1 if traj.terminal else 0
        true_rews = traj.rews

        # This is just so we can get values for all observations, the added act is not used
        # The other option is to filter out last obs here and then manually calculate and append the
        # modified_acts = torch.cat((acts,acts[..., -1:]),dim=-1)
        values, log_forwards, entropy = self.forward_policy.evaluate_actions(obs[..., :-1,:], acts)

        _, log_backwards, _ = self.backward_policy.evaluate_actions(obs[..., 1:, :], acts)

        log_Z = torch.log(self.Z_param + 1e-8)

        rewards, dis_rewards = self.reward_net.forward_trajectory(obs, acts, done, values)

        # Reward - Value loss:
        discounts = torch.full_like(rewards, self.gamma)
        discounts[..., 0] = 1
        discounts = discounts.cumprod(dim=-1)

        disc_rew = discounts * rewards
        disc_val = discounts * values.squeeze()

        # TODO: if max_ent then should we include entropy?
        val_target = disc_rew.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        value_loss = (disc_val - val_target).pow(2).mean()

        # Balance Loss:
        log_forward = log_forwards.cumsum(dim=-1)
        log_backward = log_backwards.cumsum(dim=-1)
        reward = rewards.cumsum(dim=-1)
        # would returns make more sense?
        balance_loss = (log_Z + log_forward - reward - log_backward).pow(2).mean()

        # Total Loss:
        loss = balance_loss + value_loss * self.val_coef

        reward_advantage_correl = self.get_correl(rewards, true_rews)
        reward_correl = self.get_correl(dis_rewards, true_rews)
        stats = {
            "gflow/balance_loss": balance_loss.item(),
            "gflow/value_loss": value_loss.item(),
            "gflow/reward_correl": reward_correl,
            "gflow/reward_advantage_correl": reward_advantage_correl,
            "gflow/loss": loss.item()
        }
        return loss, stats

    def log(self, losses):
        for k, v in losses.items():
            self.custom_logger.record(k, v)

    def get_correl(self, a, b):
        # not set up for batch dims.
        #flatten?
        if not isinstance(a, np.ndarray):
            a = a.squeeze().detach().cpu().numpy()
        if not isinstance(b, np.ndarray):
            b = b.squeeze().detach().cpu().numpy()
        if b.std()==0:
            return np.nan
        assert(len(a) == len(b))
        return np.corrcoef(a, b)[0,1]

