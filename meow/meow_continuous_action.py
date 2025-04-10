# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# try:
#     from cleanrl.nf.nets import MLP
#     from cleanrl.nf.transforms import Preprocessing
#     from cleanrl.nf.distributions import ConditionalDiagLinearGaussian
#     from cleanrl.nf.flows import MaskedCondAffineFlow, CondScaling
# except:
from .nf.nets import MLP
from .nf.transforms import Preprocessing
from .nf.distributions import ConditionalDiagLinearGaussian
from .nf.flows import MaskedCondAffineFlow, CondScaling


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "EfficientIRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    description: str = ""
    grad_clip: float = 30
    sigma_max: float = -0.3
    sigma_min: float = -5.0
    deterministic_action: bool = True


def evaluate(envs, policy, deterministic=True, device='cuda'):
    with torch.no_grad():
        policy.eval()
        num_envs = envs.unwrapped.num_envs
        rewards = np.zeros((num_envs,))
        dones = np.zeros((num_envs,)).astype(bool)
        s, _ = envs.reset(seed=range(num_envs))
        while not all(dones):
            a, _ = policy.sample(num_samples=s.shape[0], obs=s, deterministic=deterministic)
            a = a.cpu().detach().numpy()
            s_, r, terminated, truncated, _ = envs.step(a)
            done = terminated | truncated
            rewards += r * (1 - dones)
            dones |= done
            s = s_
    return rewards.mean()


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        env.action_space.seed(seed)
        return env

    return thunk


def init_Flow(sigma_max, sigma_min, action_sizes, state_sizes, hidden_layers=2, hidden_sizes=64, flow_layers=2,
              scale_hidden_sizes=256):
    init_parameter = "zero"
    init_parameter_flow = "orthogonal"
    dropout_rate_flow = 0.1
    dropout_rate_scale = 0.0
    layer_norm_flow = True
    layer_norm_scale = False

    # Construct the prior distribution and the linear transformation
    prior_list = [state_sizes] + [hidden_sizes] * hidden_layers + [action_sizes]
    loc = None
    log_scale = MLP(prior_list, init=init_parameter)
    q0 = ConditionalDiagLinearGaussian(action_sizes, loc, log_scale, SIGMA_MIN=sigma_min, SIGMA_MAX=sigma_max)

    # Construct normalizing flow
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_sizes)])
    for i in range(flow_layers):
        layers_list = [action_sizes + state_sizes] + [hidden_sizes] * hidden_layers + [action_sizes]
        s = None
        t1 = MLP(layers_list, init=init_parameter_flow, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        t2 = MLP(layers_list, init=init_parameter_flow, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        flows += [MaskedCondAffineFlow(b, t1, s)]
        flows += [MaskedCondAffineFlow(1 - b, t2, s)]

    # Construct the reward shifting function
    scale_list = [state_sizes] + [scale_hidden_sizes] * hidden_layers + [1]
    learnable_scale_1 = MLP(scale_list, init=init_parameter, dropout_rate=dropout_rate_scale,
                            layernorm=layer_norm_scale)
    learnable_scale_2 = MLP(scale_list, init=init_parameter, dropout_rate=dropout_rate_scale,
                            layernorm=layer_norm_scale)
    flows += [CondScaling(learnable_scale_1, learnable_scale_2)]

    # Construct the preprocessing layer
    flows += [Preprocessing()]
    return flows, q0


class HybridPolicy(nn.Module):
    def __init__(self, alpha, sigma_max, sigma_min, action_sizes, state_sizes, device,
                 hidden_layers=0, hidden_sizes=16, flow_layers=1,
                 scale_hidden_sizes=16, latent_sizes=8, mlp_hid_sizes=256, mlp_hid_layers=4):
        super(HybridPolicy, self).__init__()
        layers_list = [state_sizes] + [mlp_hid_sizes] * mlp_hid_layers + [latent_sizes]
        self.encoder = MLP(layers_list, init="orthogonal")
        self.flow_policy = FlowPolicy(alpha, sigma_max, sigma_min, action_sizes, latent_sizes, device,
                                      hidden_layers, hidden_sizes, flow_layers, scale_hidden_sizes)

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        latents = self.encode(obs)
        return self.flow_policy.predict(latents, state, episode_start, deterministic)

    def forward(self, obs, act):
        latents = self.encode(obs)
        return self.flow_policy.forward(latents, act)

    # maybe we don't need this decorator?
    @torch.jit.export
    def inverse(self, obs, act):
        latents = self.encode(obs)
        return self.flow_policy.inverse(latents, act)

    @torch.jit.ignore
    def sample(self, num_samples, obs, deterministic=False):
        latents = self.encode(obs)
        return self.flow_policy.sample(num_samples, latents, deterministic)

    @torch.jit.export
    def log_prob(self, obs, act):
        latents = self.encode(obs)
        return self.flow_policy.log_prob(latents, act)

    @torch.jit.export
    def get_qv(self, obs, act):
        latents = self.encode(obs)
        return self.flow_policy.get_qv(latents, act)

    @torch.jit.export
    def get_v(self, obs):
        latents = self.encode(obs)
        return self.flow_policy.get_v(latents)

    def entropy(self, obs, num_samples=10):
        latents = self.encode(obs)
        return self.flow_policy.entropy(latents, num_samples)


class FlowPolicy(nn.Module):
    def __init__(self, alpha, sigma_max, sigma_min, action_sizes, state_sizes, device,
                 hidden_layers=2, hidden_sizes=64, flow_layers=2,
                 scale_hidden_sizes=256
                 ):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.action_shape = action_sizes
        flows, q0 = init_Flow(sigma_max, sigma_min, action_sizes, state_sizes,
                              hidden_layers, hidden_sizes, flow_layers, scale_hidden_sizes)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.prior = q0.to(self.device)

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        with torch.no_grad():
            a, _ = self.sample(num_samples=obs.shape[0], obs=obs, deterministic=deterministic)
        return a.cpu().numpy(), state

    def forward(self, obs, act):
        log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
        z = act
        for flow in self.flows:
            z, log_det = flow.forward(z, context=obs)
            log_q -= log_det
        return z, log_q

    @torch.jit.export
    def inverse(self, obs, act):
        log_q = torch.zeros(act.shape[0], dtype=act.dtype, device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, log_det = flow.inverse(z, context=obs)
            log_q += log_det
        return z, log_q

    @torch.jit.ignore
    def sample(self, num_samples, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:  # (Warning: This is only implemented for MEow with the additive coupling layers and the Gaussian prior)
            eps = torch.randn((num_samples,) + self.prior.shape, dtype=obs.dtype, device=obs.device)
            act, _ = self.prior.get_mean_std(eps, context=obs)
            log_q = self.prior.log_prob(act, context=obs)
        else:
            act, log_q = self.prior.sample(num_samples=num_samples, context=obs)
        a, log_det = self.forward(obs=obs, act=act)
        log_q -= log_det
        return a, log_q

    @torch.jit.export
    def log_prob(self, obs, act):
        z, log_q = self.inverse(obs=obs, act=act)
        log_q += self.prior.log_prob(z, context=obs)
        return log_q

    @torch.jit.export
    def get_qv(self, obs, act):
        q = torch.zeros((act.shape[0]), device=act.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, q_, v_ = flow.get_qv(z, context=obs)
            q += q_
            v += v_
        q_, v_ = self.prior.get_qv(z, context=obs)
        q += q_
        v += v_
        q = q * self.alpha
        v = v * self.alpha
        return q[:, None], v[:, None]

    @torch.jit.export
    def get_v(self, obs):
        act = torch.zeros((obs.shape[0], self.action_shape), device=self.device)
        v = torch.zeros((act.shape[0]), device=act.device)
        z = act
        for flow in self.flows[::-1]:
            z, _, v_ = flow.get_qv(z, context=obs)
            v += v_
        _, v_ = self.prior.get_qv(z, context=obs)
        v += v_
        v = v * self.alpha
        return v[:, None]

    def entropy(self, obs, num_samples=10):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # shape: (1, state_dim)

        # Sample multiple actions from the policy.
        # The sample function returns actions and their log probabilities.
        actions = []
        for _ in range(num_samples):
            acts, _ = self.sample(num_samples=obs.shape[0], obs=obs, deterministic=False)
            actions.append(acts)

        # Ensure the state is repeated to match the number of sampled actions.
        obs_rep = obs.repeat(num_samples, 1)
        # state_rep = state.repeat(actions.shape[0], 1)

        # Compute Q-values for the (state, action) pairs.
        Q, _ = self.get_qv(obs=obs_rep, act=torch.cat(actions, 0))
        Q_mean = Q.reshape((num_samples, obs.shape[0], -1)).mean(dim=0)

        # Get the soft value V(s); get_v() returns a tensor of shape (batch_size, 1)
        V = self.get_v(obs)

        # The entropy of the policy is given by (V(s) - E[Q(s,a)])/alpha.
        entropy = (V - Q_mean) / self.alpha
        return entropy


def compute_policy_entropy(policy, obs, num_samples=20):
    """
    Computes the entropy of the MEow policy for a given state.

    Args:
        policy (FlowPolicy): the policy instance.
        obs (np.ndarray or torch.Tensor): state vector (or batch of states).
        num_samples (int): number of action samples to draw.

    Returns:
        float: entropy estimate.
    """
    # Convert state to a tensor with batch dimension if necessary.
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32, device=policy.device)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)  # shape: (1, state_dim)

    # Sample multiple actions from the policy.
    # The sample function returns actions and their log probabilities.
    actions = []
    for _ in range(num_samples):
        acts, _ = policy.sample(num_samples=obs.shape[0], obs=obs, deterministic=False)
        actions.append(acts)

    # Ensure the state is repeated to match the number of sampled actions.
    obs_rep = obs.repeat(num_samples, 1)
    # state_rep = state.repeat(actions.shape[0], 1)

    # Compute Q-values for the (state, action) pairs.
    Q, _ = policy.get_qv(obs=obs_rep, act=torch.cat(actions, 0))
    Q_mean = Q.reshape((num_samples, obs.shape[0], -1)).mean(dim=0)

    # Get the soft value V(s); get_v() returns a tensor of shape (batch_size, 1)
    V = policy.get_v(obs)

    # The entropy of the policy is given by (V(s) - E[Q(s,a)])/alpha.
    entropy = (V - Q_mean) / policy.alpha
    return entropy


def train(args=None):
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )

    if args is not None:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = SummaryWriter(args.description)
    else:
        args = tyro.cli(Args)
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        args.description = f"runs/{run_name}"
        writer = SummaryWriter(f"runs/{run_name}")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Setup the training and testing environment
    n_envs = 10
    envs, test_envs = create_envs_meow(args.env_id, args.seed, n_envs, run_name, args.capture_video)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    policy = FlowPolicy(alpha=args.alpha,
                        sigma_max=args.sigma_max,
                        sigma_min=args.sigma_min,
                        action_sizes=envs.action_space.shape[1],
                        state_sizes=envs.observation_space.shape[1],
                        device=device).to(device)
    policy_target = FlowPolicy(alpha=args.alpha,
                               sigma_max=args.sigma_max,
                               sigma_min=args.sigma_min,
                               action_sizes=envs.action_space.shape[1],
                               state_sizes=envs.observation_space.shape[1],
                               device=device).to(device)
    policy_target.load_state_dict(policy.state_dict())
    q_optimizer = optim.Adam(policy.parameters(), lr=args.q_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    best_test_rewards = -np.inf
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            policy.eval()
            actions, _ = policy.sample(num_samples=obs.shape[0], obs=obs, deterministic=False)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                # I don't think you need to do this, because when dones is true, we don't use the next val, and have zero instead.
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                # speed up training by removing true_v and min_q
                policy_target.eval()
                v_old = policy_target.get_v(torch.cat((data.next_observations, data.next_observations), dim=0))
                exact_v_old = torch.min(v_old[:v_old.shape[0] // 2], v_old[v_old.shape[0] // 2:])
                target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (exact_v_old).view(-1)

            policy.train()  # for dropout
            current_q1, _ = policy.get_qv(torch.cat((data.observations, data.observations), dim=0),
                                          torch.cat((data.actions, data.actions), dim=0))
            target_q = torch.cat((target_q, target_q), dim=0)
            qf_loss = F.mse_loss(current_q1.flatten(), target_q.flatten())
            qf_loss[qf_loss != qf_loss] = 0.0
            qf_loss = qf_loss.mean()

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    if args.autotune:
                        with torch.no_grad():
                            policy.eval()  # for dropout
                            _, log_pi = policy.sample(num_samples=data.observations.shape[0], obs=data.observations,
                                                      deterministic=args.deterministic_action)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(policy.parameters(), policy_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            if global_step % 10000 == 0:
                test_rewards = evaluate(test_envs, policy, deterministic=args.deterministic_action, device=device)
                writer.add_scalar("Test/return", test_rewards, global_step)
                writer.add_scalar("Steps", global_step, global_step)
                if test_rewards > best_test_rewards:
                    best_test_rewards = test_rewards
                    torch.save(policy, os.path.join(f"{args.description}", 'test_rewards.pt'))
                    print(
                        f"save agent to: {args.description} with best return {best_test_rewards} at step {global_step}")

    envs.close()
    writer.close()


def create_envs_meow(env_id, seed, n_envs, run_name=None, capture_video=False):
    # TODO: train in more envs???
    if capture_video and run_name is None:
        raise Exception(f"run_name must be provided if capturing video")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, capture_video, run_name)])
    test_envs = gym.make_vec(env_id, num_envs=n_envs)
    test_envs = gym.wrappers.RescaleAction(test_envs, min_action=-1.0, max_action=1.0)
    return envs, test_envs


class MEOW:
    def __init__(
            self,
            envs,
            test_envs,
            policy=None,
            alpha=0.2,
            sigma_max=-.3,
            sigma_min=-5.,
            device=None,
            q_lr=1e-3,
            autotune=False,
            seed=1,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            target_network_frequency=1,
            deterministic_action=True,
            policy_frequency=2,
            grad_clip=30,
            buffer_size=int(1e6),
            logdir=None,
            evaluate=None,
            policy_constructor:Callable=FlowPolicy,
    ):
        self.evaluate = evaluate
        self.logdir = logdir
        assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy_constructor(
            alpha=alpha,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            action_sizes=envs.action_space.shape[-1],
            state_sizes=envs.observation_space.shape[-1],
            device=device).to(device)
        self.policy_target = policy_constructor(
            alpha=alpha,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            action_sizes=envs.action_space.shape[-1],
            state_sizes=envs.observation_space.shape[-1],
            device=device).to(device)
        self.warm_start = False
        if policy is not None:
            self.policy.load_state_dict(policy.state_dict())
            self.warm_start = True
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.q_optimizer = optim.Adam(self.policy.parameters(), lr=q_lr)
        self.a_optimizer = None
        # Automatic entropy tuning
        self.alpha = alpha
        if autotune:
            self.target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)

        envs.observation_space.dtype = np.float32

        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.deterministic_action = deterministic_action
        self.autotune = autotune
        self.policy_frequency = policy_frequency
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.envs = envs
        self.test_envs = test_envs
        self.buffer_size = buffer_size

    def learn(self, total_timesteps, learning_starts=5000, wandb=None, callback=None):
        if callback:
            callback.cust_training_start(locals(), globals(), self.envs.num_envs)

        rb = ReplayBuffer(
            self.buffer_size,
            self.envs.observation_space,
            self.envs.action_space,
            self.device,
            n_envs=self.envs.num_envs,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        best_test_rewards = -np.inf
        # TRY NOT TO MODIFY: start the game
        obs = self.envs.reset()
        for global_step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < learning_starts and not self.warm_start:
                actions = np.array([self.envs.action_space.sample() for _ in range(self.envs.num_self.envs)])
            else:
                self.policy.eval()
                actions, _ = self.policy.sample(num_samples=obs.shape[0], obs=obs, deterministic=False)
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = self.envs.step(actions)
            # maybe need to put dones here instead
            if callback:
                callback.update_locals(locals())
                callback.custom_step(global_step)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            ep_stats = [info["episode"] for info in infos if "episode" in info]
            if len(ep_stats) > 0:
                ep_rew = [info["r"] for info in ep_stats]
                print(f"global_step={global_step}, mean_episodic_return={np.mean(ep_rew):.2f}")
                wandb.log({
                    "charts/global_step": global_step,
                    "charts/mean_episodic_return": np.mean(ep_rew),
                    "charts/std_episodic_return": np.std(ep_rew),
                    "charts/mean_episodic_length": np.mean([info["l"] for info in ep_stats])
                })

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            # for idx, trunc in enumerate(truncations):
            #     if trunc:
            #         real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > learning_starts:
                data = rb.sample(self.batch_size)
                with torch.no_grad():
                    # speed up training by removing true_v and min_q
                    self.policy_target.eval()
                    v_old = self.policy_target.get_v(torch.cat((data.next_observations, data.next_observations), dim=0))
                    exact_v_old = torch.min(v_old[:v_old.shape[0] // 2], v_old[v_old.shape[0] // 2:])
                    target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (exact_v_old).view(-1)

                self.policy.train()  # for dropout
                current_q1, _ = self.policy.get_qv(torch.cat((data.observations, data.observations), dim=0),
                                                   torch.cat((data.actions, data.actions), dim=0))
                target_q = torch.cat((target_q, target_q), dim=0)
                qf_loss = F.mse_loss(current_q1.flatten(), target_q.flatten())
                qf_loss[qf_loss != qf_loss] = 0.0
                qf_loss = qf_loss.mean()

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.q_optimizer.step()

                if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                            self.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        if self.autotune:
                            with torch.no_grad():
                                self.policy.eval()  # for dropout
                                _, log_pi = self.policy.sample(num_samples=data.observations.shape[0],
                                                               obs=data.observations,
                                                               deterministic=self.deterministic_action)
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.target_network_frequency == 0:
                    for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % 1000 == 0:
                    logs = {
                        "losses/global_step": global_step,
                        "losses/qf_loss": qf_loss.item(),
                        "losses/alpha": self.alpha,
                        "charts/global_step": global_step,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    }
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    if self.autotune:
                        logs.update({
                            "losses/alpha_loss": alpha_loss.item()
                        })
                    wandb.log(logs)

                if global_step % 10000 == 0:
                    test_rewards, per_expert, std_err = self.evaluate(self.test_envs, self)
                    if test_rewards > best_test_rewards:
                        best_test_rewards = test_rewards
                        torch.save(self.policy, os.path.join(f"{self.logdir}", 'test_rewards.pt'))
                        print(
                            f"save agent to: {self.logdir} with best return {best_test_rewards} at step {global_step}")


# def train_class():
#     import wandb
#     env_name = "seals:seals/Hopper-v1"
#     n_envs = 8
#     norm_reward = False
#     seed = 42
#     wandb_login()
#
#     envs, test_envs = create_envs_meow_imitation_compat(env_name, n_envs, norm_reward, seed)
#     learner = MEOW(envs, test_envs, policy_constructor=HybridPolicy)
#     learner.learn(100_000, wandb=wandb)


if __name__ == '__main__':
    train()
