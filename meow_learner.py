from helper_local import create_envs_meow_imitation_compat
from meow.meow_continuous_action import HybridPolicy, MEOW, create_envs_meow
from private_login import wandb_login


def train_class():
    import wandb
    # env_name = "seals:seals/Hopper-v1"
    env_name = "Hopper-v4"
    n_envs = 8
    norm_reward = False
    seed = 42
    wandb_login()

    # envs, test_envs = create_envs_meow_imitation_compat(env_name, n_envs, norm_reward, seed)
    envs, test_envs = create_envs_meow(env_name, seed, n_envs)
    learner = MEOW(envs, test_envs, policy_constructor=HybridPolicy)
    learner.learn(100_000, wandb=wandb)

if __name__ == "__main__":
    train_class()
