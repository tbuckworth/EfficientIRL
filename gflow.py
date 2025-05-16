import numpy as np
import gymnasium as gym
import torch
from imitation.rewards.reward_nets import BasicRewardNet, ShapedRewardNet, BasicPotentialMLP
from matplotlib import pyplot as plt
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
                 reward_type="state-action",
                 device=None,
                 val_coef=1.,
                 hard=True,
                 use_returns=True,
                 use_z=True,
                 kl_coef=1.,
                 adv_coef=0.,
                 log_prob_loss=None,
                 target_log_probs=False,
                 target_back_probs=False,
                 use_scheduler=False,
                 n_epochs=None,
                 value_is_potential=True,
                 ):
        assert not (n_epochs is None and use_scheduler), "use_scheduler requires n_epochs (predicted total training epochs"
        self.value_is_potential = value_is_potential
        self.n_epochs = n_epochs
        self.use_scheduler = use_scheduler
        self.stats = {}
        self.reward_type = reward_type
        self.target_log_probs = target_log_probs
        self.log_prob_loss = log_prob_loss
        self.target_back_probs = target_back_probs
        self.kl_coef = kl_coef
        self.adv_coef = adv_coef
        self.use_z = use_z
        self.use_returns = use_returns
        self.hard = hard
        self.val_coef = val_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net_arch is None:
            net_arch = [256, 256, 256, 256]
        self.observation_space = observation_space
        self.action_space = action_space
        self.one_hot = isinstance(self.action_space, gym.spaces.Discrete)
        self.demonstrations = demonstrations
        self.gamma = gamma
        self.batch_size = batch_size
        self.l2_weight = l2_weight
        # default seed if rng is not supplied
        self.rng = rng or np.random.default_rng(0)
        self.custom_logger = custom_logger

        self.Z_param = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_policy = get_policy_for(observation_space, action_space, net_arch)
        self.backward_policy = get_policy_for(observation_space, action_space, net_arch)

        value_func = lambda obs: self.forward_policy.predict_values(obs).squeeze()

        use_state = use_action = use_next_state = False
        if reward_type == "state-action":
            use_state = use_action = True
        if reward_type == "state":
            use_state = True
        if reward_type in ["next state", "next state only"]:
            use_next_state = True

        self.output_reward_function = BasicRewardNet(
            observation_space,
            action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state
        )

        nets = [self.forward_policy, self.backward_policy, self.output_reward_function, self.Z_param]
        param_list = list(self.forward_policy.parameters()) + list(self.backward_policy.parameters()) + list(
            self.output_reward_function.parameters()) + [self.Z_param]
        if self.value_is_potential:
            pot_net = value_func
        else:
            pot_net = BasicPotentialMLP(
                observation_space=observation_space,
                hid_sizes=net_arch,
            )
            nets += [pot_net]
            param_list += list(pot_net.parameters())
        self.reward_net = CustomShapedRewardNet(
            self.output_reward_function,
            pot_net,
            gamma)
        _ = [n.to(device=self.device) for n in nets]

        self.optimizer = torch.optim.Adam(param_list,
                                          lr=lr)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs,
                                                                        eta_min=0)
        if self.log_prob_loss == "kl":
            self.maybe_optimize_log_probs = lambda lf: -lf.mean()
        elif self.log_prob_loss == "chi_square":
            self.maybe_optimize_log_probs = lambda lf: -(lf - (2 * lf.pow(2))).mean()
        elif self.log_prob_loss is None:
            if self.target_log_probs:
                raise Exception("forward policy won't learn if log_prob_loss is None and target_log_probs is True")
            self.maybe_optimize_log_probs = lambda lf: 0
        elif self.log_prob_loss == "abs":
            self.maybe_optimize_log_probs = lambda lf: lf.abs().mean()
        else:
            raise NotImplementedError(f"{self.log_prob_loss} is not a valid log_prob_loss")

    @property
    def policy(self) -> ActorCriticPolicy:
        return self.forward_policy

    @property
    def reward_func(self) -> BasicRewardNet:
        return self.output_reward_function

    @property
    def state_reward_func(self) -> BasicRewardNet:
        return self.output_reward_function

    def train(self, n_epochs, progress_bar=None, log=True, split_training=None):
        assert split_training is None or (0 < split_training < 1), "split_training must be between 0 and 1 or None"
        for epoch in range(n_epochs):
            if split_training is None:
                loss_calc = self.compute_all_losses
            elif epoch < split_training*n_epochs:
                loss_calc = self.compute_log_prob_losses
            else:
                loss_calc = self.compute_trajectory_balance_losses

            if log and self.custom_logger is not None:
                self.custom_logger.record("epoch", epoch)
            for traj in self.rng.permutation(self.demonstrations):
                loss, stats = loss_calc(traj)
                loss.backward()
                self.optimizer.step()
                if self.use_scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.accum_stats(stats)
            if log:
                self.log(epoch)

    def compute_all_losses(self, traj):
        obs = torch.FloatTensor(traj.obs).to(device=self.device)
        acts = torch.FloatTensor(traj.acts).to(device=self.device)
        infos = traj.infos
        done = torch.zeros_like(acts).to(device=self.device)
        # if we batch trajs, then will have to modify this
        done[-1] = 1 if traj.terminal else 0
        true_rews = traj.rews

        # obs has one extra observation (terminal)
        values, log_forwards, entropy = self.forward_policy.evaluate_actions(obs[..., :-1, :], acts)
        _, log_backwards, _ = self.backward_policy.evaluate_actions(obs[..., 1:, :], acts)
        log_Z = torch.log(self.Z_param + 1e-8) if self.use_z else 0
        rew_acts = self.maybe_one_hot(acts)
        rewards, dis_rewards = self.reward_net.forward_trajectory(obs, rew_acts, done, values)

        # Reward - Value loss:
        discounts = torch.full_like(rewards, self.gamma)
        discounts[..., 0] = 1
        discounts = discounts.cumprod(dim=-1)

        disc_rew = discounts * (rewards + (entropy.squeeze() if not self.hard else 0))
        disc_val = discounts * values.squeeze()

        val_target = disc_rew.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        value_loss = ((disc_val - val_target) / discounts).pow(2).mean()

        kl_loss = self.maybe_optimize_log_probs(log_forwards)
        if self.target_back_probs:
            kl_loss = kl_loss + self.maybe_optimize_log_probs(log_backwards)

        # Part of me thinks we should not have the entropy here or above. But that option is not accounted for.
        # Also should we use actor_adv.detach() instead? I think the balance may be good for the actor, but not sure.
        actor_adv = log_forwards + (entropy.squeeze() if self.hard else 0)
        adv_loss = (actor_adv - rewards).pow(2).mean()

        # Balance Loss:
        if self.target_log_probs:
            log_forwards = log_forwards.detach()
            if self.target_back_probs:
                log_backwards = log_backwards.detach()

        log_forward = log_forwards.cumsum(dim=-1)
        log_backward = log_backwards.cumsum(dim=-1)
        if self.use_returns:
            reward = disc_rew.cumsum(dim=-1)
        else:
            reward = rewards.cumsum(dim=-1)
        balance_loss = (log_Z + log_forward - reward - log_backward).pow(2).mean()

        # Total Loss:
        loss = balance_loss + value_loss * self.val_coef + kl_loss * self.kl_coef + adv_loss * self.adv_coef

        reward_advantage_correl = self.get_correl(rewards, true_rews)
        reward_correl = self.get_correl(dis_rewards, true_rews)
        if not np.isnan(reward_correl) and reward_correl > 0.95:
            pass
        if isinstance(kl_loss, int):
            kl = np.nan
        else:
            kl = kl_loss.item()
        stats = {
            "gflow/balance_loss": balance_loss.item(),
            "gflow/value_loss": value_loss.item(),
            "gflow/kl_loss": kl,
            "gflow/reward_correl": reward_correl,
            "gflow/reward_advantage_correl": reward_advantage_correl,
            "gflow/loss": loss.item()
        }
        return loss, stats

    def compute_log_prob_losses(self, traj):
        obs = torch.FloatTensor(traj.obs).to(device=self.device)
        acts = torch.FloatTensor(traj.acts).to(device=self.device)
        done = torch.zeros_like(acts).to(device=self.device)
        # if we batch trajs, then will have to modify this
        done[-1] = 1 if traj.terminal else 0

        # obs has one extra observation (terminal)
        _, log_forwards, _ = self.forward_policy.evaluate_actions(obs[..., :-1, :], acts)
        _, log_backwards, _ = self.backward_policy.evaluate_actions(obs[..., 1:, :], acts)

        kl_loss = self.maybe_optimize_log_probs(log_forwards)
        if self.target_back_probs:
            kl_loss = kl_loss + self.maybe_optimize_log_probs(log_backwards)

        stats = {
            "gflow/kl_loss": kl_loss.item(),
        }
        return kl_loss, stats

    def compute_trajectory_balance_losses(self, traj):
        assert not self.value_is_potential, "reward potential net must not be policy value function if training kl and traj balance separately.\nSet value_is_potential=False in GFLOW init."
        obs = torch.FloatTensor(traj.obs).to(device=self.device)
        acts = torch.FloatTensor(traj.acts).to(device=self.device)
        infos = traj.infos
        done = torch.zeros_like(acts).to(device=self.device)
        # if we batch trajs, then will have to modify this
        done[-1] = 1 if traj.terminal else 0
        true_rews = traj.rews

        # obs has one extra observation (terminal)
        with torch.no_grad():
            _, log_forwards, _ = self.forward_policy.evaluate_actions(obs[..., :-1, :], acts)
            _, log_backwards, _ = self.backward_policy.evaluate_actions(obs[..., 1:, :], acts)
        log_Z = torch.log(self.Z_param + 1e-8) if self.use_z else 0
        rew_acts = self.maybe_one_hot(acts)
        rewards, dis_rewards = self.reward_net.forward_trajectory(obs, rew_acts, done, None)

        log_forward = log_forwards.cumsum(dim=-1)
        log_backward = log_backwards.cumsum(dim=-1)
        if self.use_returns:
            discounts = torch.full_like(rewards, self.gamma)
            discounts[..., 0] = 1
            discounts = discounts.cumprod(dim=-1)
            disc_rew = discounts * rewards
            reward = disc_rew.cumsum(dim=-1)
        else:
            reward = rewards.cumsum(dim=-1)
        balance_loss = (log_Z + log_forward - reward - log_backward).pow(2).mean()

        reward_advantage_correl = self.get_correl(rewards, true_rews)
        reward_correl = self.get_correl(dis_rewards, true_rews)
        if not np.isnan(reward_correl) and reward_correl > 0.95:
            pass
        stats = {
            "gflow/balance_loss": balance_loss.item(),
            "gflow/reward_correl": reward_correl,
            "gflow/reward_advantage_correl": reward_advantage_correl,
        }
        return balance_loss, stats

    def maybe_one_hot(self, acts):
        if self.one_hot:
            return torch.nn.functional.one_hot(acts.to(torch.int64), self.action_space.n).to(device=self.policy.device)
        return acts

    def accum_stats(self, stats):
        if self.stats == {}:
            self.stats = {k: [v] for k, v in stats.items()}
        else:
            [self.stats[k].append(v) for k, v in stats.items()]

    def log(self, epoch):
        for k, v in self.stats.items():
            mean_v = np.mean(np.array(v)[~np.isnan(v)])
            if self.custom_logger is not None:
                self.custom_logger.record(k, mean_v)
        if self.custom_logger is not None:
            self.custom_logger.dump(epoch)
        else:
            print(f"Epoch:{epoch}")
            for k, v in self.stats.items():
                mean_v = np.mean(np.array(v)[~np.isnan(v)])
                print(f"\t{k}: {mean_v:.2f}")
        self.stats = {}

    def get_correl(self, a, b, plot=False):
        # not set up for batch dims.
        # flatten?
        if not isinstance(a, np.ndarray):
            a = a.squeeze().detach().cpu().numpy()
        if not isinstance(b, np.ndarray):
            b = b.squeeze().detach().cpu().numpy()
        if b.std() == 0:
            return np.nan
        assert (len(a) == len(b))
        if plot:
            # plt.scatter(x=b,y=b,label="True Rewards",color="black")
            plt.scatter(x=b, y=a, label="Learned Rewards")
            plt.legend()
            plt.show()
        return np.corrcoef(a, b)[0, 1]
