import os
import numpy as np
import torch
from imitation.util import logger as imit_logger
from stable_baselines3 import TD3
from helper_local import import_wandb, create_logdir, load_env

wandb = import_wandb()

def trainRL(
            seed=42,
            learner_timesteps=5000_000,
            n_envs=16,
            extra_tags=None,
            env_name="seals:seals/Hopper-v1",
            ):
    algo = "TD3"
    tags = [] + (extra_tags if extra_tags is not None else [])
    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), locals())
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=locals(), tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])

    default_rng, env = load_env(env_name, n_envs, seed, norm_reward=False)

    learner = TD3('MlpPolicy',
                  env,
                  learning_rate=0.0003,
                  batch_size=256,
                  gradient_steps=1,
                  learning_starts=10000,
                  train_freq=1,
                  tensorboard_log=logdir,
                  )

    learner.learn(learner_timesteps)
    torch.save({'model_state_dict': learner.policy.state_dict()},
               f'{logdir}/model_RL_{learner_timesteps}.pth')
    wandb.finish()



if __name__ == "__main__":
    trainRL(extra_tags=["TD3","Train","RL"])
