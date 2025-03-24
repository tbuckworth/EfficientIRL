from collections import OrderedDict

import torch
from imitation.policies.base import NormalizeFeaturesExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.sac.policies import SACPolicy


def load_ant_ppo_learner(env, logdir, policy):
    antv1_params = OrderedDict([('batch_size', 16),
                                ('clip_range', 0.3),
                                ('ent_coef', 3.1441389214159857e-06),
                                ('gae_lambda', 0.8),
                                ('gamma', 0.995),
                                ('learning_rate', 0.00017959211641976886),
                                ('max_grad_norm', 0.9),
                                ('n_epochs', 10),
                                ('n_steps', 2048),
                                # ('n_timesteps', 1000000.0),
                                ('normalize',
                                 {'gamma': 0.995, 'norm_obs': False, 'norm_reward': True}),
                                ('policy', 'MlpPolicy'),
                                ('policy_kwargs',
                                 {'activation_fn': torch.nn.modules.activation.Tanh,
                                  'features_extractor_class': NormalizeFeaturesExtractor,
                                  'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
                                ('vf_coef', 0.4351450387648799),
                                ('normalize_kwargs',
                                 {'norm_obs': {'gamma': 0.995,
                                               'norm_obs': False,
                                               'norm_reward': True},
                                  'norm_reward': False})])
    ant_params = dict(
        policy='MlpPolicy',
        env=env,
        learning_rate=0.00017959211641976886,
        n_steps=2048,
        batch_size=16,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.8,
        clip_range=0.3,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=3.1441389214159857e-06,
        vf_coef=0.4351450387648799,
        max_grad_norm=0.9,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=logdir,
        policy_kwargs={'activation_fn': torch.nn.modules.activation.Tanh,
                       'features_extractor_class': NormalizeFeaturesExtractor,
                       'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]},
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )
    learner = PPO(**ant_params)
    learner.policy = policy
    return learner


def load_ant_sac_learner(env, logdir, policy):
    OrderedDict([('batch_size', 512),
                 ('buffer_size', 1000000),
                 ('gamma', 0.98),
                 ('learning_rate', 0.0018514039303149058),
                 ('learning_starts', 1000),
                 ('n_timesteps', 1000000.0),
                 ('policy', 'MlpPolicy'),
                 ('policy_kwargs',
                  {'log_std_init': -2.2692589009754176,
                   'net_arch': [256, 256],
                   'use_sde': False}),
                 ('tau', 0.05),
                 ('train_freq', 64),
                 ('normalize', False)])
    params = dict(
        policy='MlpPolicy',
        env=env,
        learning_rate=0.0018514039303149058,
        buffer_size=1_000_000,  # 1e6
        learning_starts=1000,
        batch_size=512,
        tau=0.05,
        gamma=0.98,
        train_freq=64,
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        stats_window_size=100,
        tensorboard_log=logdir,
        policy_kwargs={'log_std_init': -2.2692589009754176,
                       'net_arch': policy.net_arch,
                       'use_sde': False},
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )
    learner = SAC(**params)

    # could be just features_extractor and not pi - check eirl
    learner.policy.actor.features_extractor.load_state_dict(policy.pi_features_extractor.state_dict())
    learner.policy.actor.latent_pi.load_state_dict(policy.mlp_extractor.policy_net.state_dict())
    learner.policy.actor.mu.load_state_dict(policy.action_net.state_dict())
    learner.policy.actor.log_std.load_state_dict(policy.log_std_net.state_dict())

    return learner


def load_hopper_ppo_learner(env, logdir, policy):
    # OrderedDict([('batch_size', 512),
    #              ('clip_range', 0.1),
    #              ('ent_coef', 0.0010159833764878474),
    #              ('gae_lambda', 0.98),
    #              ('gamma', 0.995),
    #              ('learning_rate', 0.0003904770450788824),
    #              ('max_grad_norm', 0.9),
    #              ('n_envs', 1),
    #              ('n_epochs', 20),
    #              ('n_steps', 2048),
    #              ('n_timesteps', 1000000.0),
    #              ('normalize',
    #               {'gamma': 0.995, 'norm_obs': False, 'norm_reward': True}),
    #              ('policy', 'MlpPolicy'),
    #              ('policy_kwargs',
    #               {'activation_fn': <

    # class 'torch.nn.modules.activation.ReLU'>,
    #
    # 'features_extractor_class': <
    #
    # class 'imitation.policies.base.NormalizeFeaturesExtractor'>,
    #
    # 'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
    # ('vf_coef', 0.20315938606555833),
    # ('normalize_kwargs',
    #  {'norm_obs': {'gamma': 0.995,
    #                'norm_obs': False,
    #                'norm_reward': True},
    # 'norm_reward': False})])
    params = dict(
        policy='MlpPolicy',
        env=env,
        learning_rate=0.0003904770450788824,
        n_steps=2048,
        batch_size=512,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.1,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0010159833764878474,
        vf_coef=0.20315938606555833,
        max_grad_norm=0.9,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=logdir,
        policy_kwargs={'activation_fn': torch.nn.modules.activation.ReLU,
                       'features_extractor_class': NormalizeFeaturesExtractor,
                       'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]},
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )
    learner = PPO(**params)
    learner.policy = policy
    return learner


def load_cartpole_ppo_learner(wenv, logdir, policy):
    OrderedDict([('batch_size', 256),
                 ('clip_range', 'lin_0.2'),
                 ('ent_coef', 0.0),
                 ('gae_lambda', 0.8),
                 ('gamma', 0.98),
                 ('learning_rate', 'lin_0.001'),
                 ('n_envs', 8),
                 ('n_epochs', 20),
                 ('n_steps', 32),
                 ('n_timesteps', 100000.0),
                 ('policy', 'MlpPolicy'),
                 ('normalize', False)])

    params = dict(
        policy='MlpPolicy',
        env=wenv,
        learning_rate=0.001,
        n_steps=32,
        batch_size=256,
        n_epochs=20,
        gamma=0.98,
        gae_lambda=0.8,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0,
        vf_coef=0.20315938606555833,
        max_grad_norm=0.9,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=logdir,
        # policy_kwargs={'activation_fn': torch.nn.modules.activation.ReLU,
        #                'features_extractor_class': NormalizeFeaturesExtractor,
        #                'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]},
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )
    learner = PPO(**params)
    learner.policy = policy
    return learner


def load_pendulum_ppo_learner(wenv, logdir, policy):
    params = dict(
        policy='MlpPolicy',
        env=wenv,
        learning_rate=0.001,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=False,  # TODO:?
        ent_coef=0,
        # vf_coef=0.20315938606555833,
        max_grad_norm=0.9,
        use_sde=False,  # not sure if best
        sde_sample_freq=4,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=logdir,
        # policy_kwargs={'activation_fn': torch.nn.modules.activation.ReLU,
        #                'features_extractor_class': NormalizeFeaturesExtractor,
        #                'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]},
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )
    learner = PPO(**params)
    learner.policy = policy
    return learner


def load_mountaincar_ppo_learner(wenv, logdir, policy):
    params = dict(
        policy='MlpPolicy',
        env=wenv,
        learning_rate=0.0004476103728105138,
        n_steps=256,
        batch_size=512,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=False,
        ent_coef=6.4940755116195606e-06,
        vf_coef=0.25988158989488963,
        max_grad_norm=1.,
        use_sde=False,  # not sure if best
        # sde_sample_freq=4,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=logdir,
        # policy_kwargs={'activation_fn': torch.nn.modules.activation.ReLU,
        #                'features_extractor_class': NormalizeFeaturesExtractor,
        #                'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]},
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )
    learner = PPO(**params)
    learner.policy = policy
    return learner


# def load_ant_sac_learner(wenv, logdir, policy):
#     learner = SAC(
#         'MlpPolicy',
#         wenv,
#         tensorboard_log=logdir,
#     )
#     learner.policy
#     return learner


def load_learner(env_name, wenv, logdir, policy, rl_algo="ppo"):
    if rl_algo == "ppo":
        if env_name == "seals:seals/Hopper-v1":
            return load_hopper_ppo_learner(wenv, logdir, policy)
        if env_name == "seals:seals/Ant-v1":
            return load_ant_ppo_learner(wenv, logdir, policy)
        if env_name == "seals:seals/CartPole-v0":
            return load_cartpole_ppo_learner(wenv, logdir, policy)
        if env_name == "Pendulum-v1":
            return load_pendulum_ppo_learner(wenv, logdir, policy)
        if env_name == "seals:seals/MountainCar-v0":
            return load_mountaincar_ppo_learner(wenv, logdir, policy)
    elif rl_algo == "sac":
        if env_name == "seals:seals/Ant-v1":
            return load_ant_sac_learner(wenv, logdir, policy)
    raise NotImplementedError
