from collections import OrderedDict

import torch
from imitation.policies.base import NormalizeFeaturesExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.sac.policies import SACPolicy


def load_ant_learner(env, logdir=None):
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
                       'net_arch': [32, 32],
                       'use_sde': False},
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    )
    learner = SAC(**params)
    learner.policy.actor = policy.actor
