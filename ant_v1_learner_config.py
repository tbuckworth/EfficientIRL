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
                 ('n_timesteps', 1000000.0),
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
    learner = PPO(env=env,**antv1_params)
    return learner
