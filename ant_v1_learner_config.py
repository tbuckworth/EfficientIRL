from collections import OrderedDict

import torch
from imitation.policies.base import NormalizeFeaturesExtractor
from stable_baselines3 import PPO


def load_ant_learner(env):
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
        tensorboard_log=None,
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
