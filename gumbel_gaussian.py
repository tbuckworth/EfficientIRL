from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.utils import seeding

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.distributions import Distribution

from helper_local import filter_params


# --------------------------------------------------------------------
# 1. Simple continuous-action environment
# --------------------------------------------------------------------
class SimpleContEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.state = np.zeros(4, dtype=np.float32)
        self.seed()  # initialize the seed

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        return self.state, {}

    def step(self, action):
        reward = -np.linalg.norm(action - 0.5)
        self.state = self.np_random.uniform(-1, 1, size=4).astype(np.float32)
        done = False
        return self.state, reward, done, done, {}

# --------------------------------------------------------------------
# 2. GumbelSoftmax → MLP → Gaussian distribution, with toggles
# --------------------------------------------------------------------
class GumbelSoftmaxGaussian(Distribution):
    """
    During training mode:
      - use soft Gumbel-Softmax + feed z into MLP => mean, log_std => sample from Gaussian
      - log_prob = log p_gumbel(z) + log p_gaussian(a)
    During eval mode:
      - use hard Gumbel-Softmax (one-hot)
      - skip the Gaussian entirely (just use the MLP’s mean as output)
      - log_prob = log p_gumbel(z) only
    """
    def __init__(self, action_dim, logit_dim, hidden_dim=64, device=None):
        super().__init__()
        self.action_dim = action_dim
        self.logit_dim = logit_dim
        # MLP that maps from "gumbel embedding" (size action_dim) -> (mean, log_std)
        self.mlp = nn.Sequential(
            nn.Linear(logit_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)  # => [mean, log_std]
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp.to(device=self.device)

        # Internal states that get set in proba_distribution(...)
        self.logits = None
        self.z_gumbel = None
        self.mean_actions = None
        self.log_std = None

        # soft => use gaussian and use soft gumbel
        self.soft = True

    def train(self, mode=True):
        """
        Toggling distribution to 'training' (soft Gumbel + Gaussian)
        or 'eval' (hard Gumbel + no Gaussian).
        """
        if mode:
            # Training mode
            self.soft = True
        else:
            # Evaluation mode
            self.soft = False
        return self

    def eval(self):
        """Just calls train(False)."""
        return self.train(False)

    def proba_distribution(self, logits: torch.Tensor) -> "GumbelSoftmaxGaussian":
        self.logits = logits
        # Hard or soft Gumbel depending on mode
        self.z_gumbel = F.gumbel_softmax(logits, tau=1.0, hard=not self.soft, dim=-1)

        # MLP => mean, log_std
        out = self.mlp(self.z_gumbel)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        self.mean_actions = mean
        self.log_std = torch.clamp(log_std, -20, 2) if self.soft else None

        return self

    def sample(self) -> torch.Tensor:
        """
        If we're in training mode: sample from the Gaussian.
        If we're in eval mode: just return the 'mean' (one-hot or a transform).
        """
        if self.soft:
            std = torch.exp(self.log_std)
            return self.mean_actions + std * torch.randn_like(std)

        # Hard, no Gaussian => just use "mean_actions"
        return self.mean_actions

    def mode(self) -> torch.Tensor:
        """
        The 'greedy' action. Usually in training we'd interpret that as the mean.
        In eval mode, it's basically the same as 'sample()' if there's no noise.
        """
        return self.mean_actions

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        log_prob = log p_gumbel(z) + [log p_gaussian(a | z) if in training mode].
        For Gumbel: we approximate with sum( z_i * log_softmax(logits)_i ).
        """
        # Gumbel part
        log_softmax_logits = F.log_softmax(self.logits, dim=-1)
        gumbel_log_prob = (self.z_gumbel * log_softmax_logits).sum(dim=-1)

        if self.soft:
            # Gaussian part
            std = torch.exp(self.log_std)
            gauss_log_prob = -0.5 * (
                ((actions - self.mean_actions) / std) ** 2
                + 2 * self.log_std
                + np.log(2.0 * np.pi)
            ).sum(dim=-1)
            return gumbel_log_prob + gauss_log_prob
        # No Gaussian => no additional log-prob
        # TODO: check then when hard, this is just the same as the log_softmax
        #  (should be, because z_gumbel should be one_hot in that case)
        return gumbel_log_prob

    def entropy(self) -> torch.Tensor:
        """
        Very approximate.
        - In training, sum of Gumbel + Gaussian entropies.
        - In eval, just Gumbel.
        """
        # Gumbel "entropy" is trickier than just -sum(z * log(z)),
        # but let's do a naive version:

        if self.soft:
            assert self.log_std is not None
            gumbel_ent = -(self.z_gumbel * F.log_softmax(self.logits, dim=-1)).sum(dim=-1)
            # Gaussian part
            gauss_ent = 0.5 * (1.0 + np.log(2 * np.pi)) * self.action_dim + self.log_std.sum(dim=-1)
            return gumbel_ent + gauss_ent
        log_probs = F.log_softmax(self.logits, dim=-1)
        entropy = -(log_probs.exp()*log_probs).sum(dim=-1)
        return entropy

    # abstract methods:

    def proba_distribution_net(self, latent_dim: int, action_space):
        """
        Create and return a network that maps latent features to distribution parameters.
        For our distribution, we simply return a linear layer that outputs logits
        for our Gumbel softmax.
        """
        net = nn.Linear(latent_dim, self.logit_dim) #should this be action dim? i don't think so
        return net

    def actions_from_params(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits, compute the distribution parameters and sample an action.
        This is equivalent to calling proba_distribution(...) then sample().
        """
        self.proba_distribution(logits)
        return self.sample()

    def log_prob_from_params(self, logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Given logits, compute the distribution parameters, sample an action, and return both
        the action and its log probability.
        """
        self.proba_distribution(logits)
        action = self.sample()
        logp = self.log_prob(action)
        return action, logp

# --------------------------------------------------------------------
# 3. Custom policy that toggles the distribution’s mode
# --------------------------------------------------------------------
class CustomGumbelPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_actions = 64
        self.logit_dim = int(np.ceil(np.log2(self.max_actions)))
        # The MlpExtractor to process observations
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=[64, 64], activation_fn=nn.ReLU)

        # The dimension for the Gumbel embedding (a "categorical" of size action_dim).
        self.action_dim = self.action_space.shape[0]
        self.logit_layer = nn.Linear(self.mlp_extractor.latent_dim_pi, self.logit_dim)

        self.custom_dist = GumbelSoftmaxGaussian(self.action_dim, self.logit_dim)

    def train(self, mode=True):
        """
        Override to ensure policy/distribution modes stay in sync.
        SB3 calls policy.train() = True for training, = False for eval.
        """
        super().train(mode)
        self.custom_dist.train(mode)
        return self

    def eval(self):
        """Calls train(False)."""
        return self.train(False)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        logits = self.logit_layer(latent_pi)
        return self.custom_dist.proba_distribution(logits)

# --------------------------------------------------------------------
# 4. Demo training loop with PPO
# --------------------------------------------------------------------
if __name__ == "__main__":
    hparams = OrderedDict([('clip_range', 0.2),
                 ('ent_coef', 0.0),
                 ('gae_lambda', 0.95),
                 ('gamma', 0.9),
                 ('learning_rate', 0.001),
                 ('n_envs', 4),
                 ('n_epochs', 10),
                 ('n_steps', 1024),
                 ('n_timesteps', 100000.0),

                 ('sde_sample_freq', 4),
                 ('use_sde', True),
                 ('normalize', False)])

    # env = make_vec_env(SimpleContEnv, n_envs=1)
    env = make_vec_env("Pendulum-v1", **filter_params(hparams, make_vec_env))
    model = PPO(
        policy=CustomGumbelPolicy,
        env=env,
        verbose=1,
        **filter_params(hparams,PPO)
    )

    # Train (soft Gumbel + Gaussian).
    model.learn(total_timesteps=1000_000)

    # Evaluate in "hard Gumbel, no Gaussian" mode.
    # SB3 does: model.policy.eval(), or we can do:
    model.policy.eval()

    rewards, _ = evaluate_policy(model.policy, env, 10, return_episode_rewards=True)
    print(f"Rewards:{rewards}")
    # obs = env.reset()
    # print("\nEVALUATION (Hard Gumbel, No Gaussian)\n")
    # for _ in range(5):
    #     # 'deterministic=False' just calls distribution.sample()
    #     # but we are in eval mode => sample() is effectively the argmax (one-hot) -> MLP -> mean.
    #     action, _ = model.predict(obs, deterministic=False)
    #     obs, reward, done, _info = env.step(action)
    #     print(f"Action: {action}, Reward: {reward}")
