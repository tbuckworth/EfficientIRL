import torch
from imitation.algorithms import base
from imitation.rewards.reward_nets import BasicShapedRewardNet, BasicRewardNet, ShapedRewardNet
from jinja2 import optimizer
from stable_baselines3.common.policies import ActorCriticPolicy

from helper_local import get_policy_for


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
                 ):
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
        self.reward_net = ShapedRewardNet(self.output_reward_function,
                                          value_func,
                                          gamma)
        self.optimizer = torch.optim.Adam(list(self.forward_policy.parameters()) +
                                          list(self.backward_policy.parameters()) +
                                          list(self.output_reward_function.parameters()) +
                                          list(self.reward_net.parameters()) +
                                          [self.Z_param],
                                          lr=lr)
        self.data = torch.utils.data.DataLoader(
            demonstrations,
            batch_size=batch_size,
            # collate_fn=types.transitions_collate_fn,

        )

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            # maybe shuffle?
            for traj in self.demonstrations:
                loss = self.trajectory_balance_loss(traj)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def trajectory_balance_loss(self, traj):
        obs = traj.obs
        acts = traj.acts
        infos = traj.infos
        terminal = traj.terminal
        true_rews = traj.rews

        probs = self.forward_policy(obs)  # [:-1] might be more informative - wouldn't change anything
        log_forwards = torch.log(probs[[i for i in range(len(acts))], acts])

        # For backward, assume a fixed reverse mapping (e.g. up <-> down, right <-> left)
        backward_action_map = {0: 2, 1: 3, 2: 0, 3: 1}
        # is this necessary?

        fwd_action = acts
        back_action = [backward_action_map[k] for k in fwd_action]
        probs = self.backward_policy(obs[1:])
        log_backwards = torch.log(probs[[i for i in range(len(probs))], back_action])

        log_Z = torch.log(self.Z_param + 1e-8)

        rewards = self.reward_net(obs).squeeze()

        log_forward = log_forwards.cumsum(dim=0)
        log_backward = log_backwards.cumsum(dim=0)
        reward = rewards[0] + rewards[1:].cumsum(dim=0)
        loss = (log_Z + log_forward - reward - log_backward).pow(2).mean()
        return loss

