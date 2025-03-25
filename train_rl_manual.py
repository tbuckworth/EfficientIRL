import os
import numpy as np
import torch
from imitation.util import logger as imit_logger
from stable_baselines3 import TD3, PPO, SAC
from helper_local import import_wandb, create_logdir, load_env, get_config, load_td3_agent
from gymnasium.wrappers import RecordVideo
import gymnasium as gym

wandb = import_wandb()

agents = {
    "TD3": TD3,
    "PPO": PPO,
    "SAC": SAC,
}

def trainRL(
            seed=42,
            learner_timesteps=500_000,
            n_envs=16,
            extra_tags=None,
            env_name="seals:seals/CartPole-v0",
            ):
    algo = "PPO"
    tags = [] + (extra_tags if extra_tags is not None else [])
    cfg = locals()
    rl_kwargs = dict(
        learning_rate=0.0003,
        batch_size=256,
        # gradient_steps=1,
        # learning_starts=10000,
        # train_freq=1,
    )
    cfg.update(rl_kwargs)

    AGENT = agents[algo]

    logdir = create_logdir(env_name, seed)
    np.save(os.path.join(logdir, "config.npy"), cfg)
    wandb.init(project="EfficientIRL", sync_tensorboard=True, config=cfg, tags=tags)
    custom_logger = imit_logger.configure(logdir, ["stdout", "csv", "tensorboard"])

    default_rng, env = load_env(env_name, n_envs, seed, norm_reward=False)

    learner = AGENT('MlpPolicy',
                  env,
                  tensorboard_log=logdir,
                  **rl_kwargs,
                  )

    learner.learn(learner_timesteps)
    torch.save({'model_state_dict': learner.policy.state_dict()},
               f'{logdir}/model_RL_{learner_timesteps}.pth')
    wandb.finish()

def record_video(logdir):
    name_prefix = "TD3"
    cfg = get_config(logdir)
    env_name = cfg["env_name"]
    n_envs = 1
    seed = cfg["seed"]
    default_rng, env = load_env(env_name, n_envs, seed, norm_reward=False)

    policy = load_td3_agent(env, logdir)

    vid_env = gym.make(env_name, render_mode="rgb_array")
    # Set up video recording parameters
    video_folder = "./videos/"
    video_length = 1000  # adjust to the desired length (in timesteps)
    # Wrap your vectorized environment with VecVideoRecorder
    vid_env = RecordVideo(
        vid_env,
        video_folder,
        episode_trigger=lambda e: True,
        video_length=video_length,
        name_prefix=name_prefix
    )

    obs, _ = vid_env.reset()
    cum_reward = 0
    for _ in range(1000):
        # Sample a random action
        with torch.no_grad():
            action = policy._predict(torch.FloatTensor(obs).to(device=policy.device).unsqueeze(0))

        # Step the environment
        obs, reward, term, trunc, info = vid_env.step(action.squeeze().cpu().numpy())
        cum_reward += reward
        # Render the *first* (index 0) sub-environment
        # This calls the 'render()' of the underlying gym environment

        # Optionally reset if done
        if term or trunc:
            print(f"Total Reward: {cum_reward}")
            vid_env.reset()
            break
    vid_env.close()


if __name__ == "__main__":
    # record_video("logs/train/seals:seals/Hopper-v1/2025-03-21__17-06-20__seed_42")
    trainRL(extra_tags=["PPO","Train","RL"])
