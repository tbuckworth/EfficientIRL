import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(env_id: str, gravity: float):
    """
    Returns a function that, when called, creates a single CartPole environment
    and sets its gravity to the specified value.
    """
    def _init():
        env = gym.make(env_id)
        # 'unwrapped' gets the actual underlying environment instance
        env.unwrapped.gravity = gravity
        return env
    return _init

def main():
    # Suppose you want 8 parallel environments with gravity=15.0:
    env_id = "CartPole-v1"
    num_envs = 8
    my_gravity = 15.0

    # Create a list of environment-building callables
    env_fns = [make_env(env_id, my_gravity) for _ in range(num_envs)]

    # Build your vectorized environment
    vec_env = DummyVecEnv(env_fns)

    agent = PPO('MlpPolicy', vec_env, verbose=1)
    agent.learn(10000)
    # From here, you can pass 'vec_env' into SB3 algorithms
    # For example:
    # from stable_baselines3 import PPO
    # model = PPO("MlpPolicy", vec_env, verbose=1)
    # model.learn(total_timesteps=10000)

if __name__ == "__main__":
    main()
