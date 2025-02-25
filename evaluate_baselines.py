import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.util import util
from imitation.scripts.train_adversarial import load_expert_trajs

from custom_irl import CustomIRL

# Load precomputed expert demonstrations from the imitation package
def load_expert_data():
    print("Loading precomputed expert demonstrations...")
    return load_expert_trajs("seals/CartPole-v0")

# Compare the performance of different methods
def compare_baselines(env, expert_demos):
    print("\nEvaluating Expert Policy...")
    expert_return = rollout.rollout_stats(expert_demos)['return_mean']

    print("\nEvaluating Custom IRL Policy...")
    custom_irl = CustomIRL(env, expert_demos)
    custom_return = custom_irl.evaluate()

    print(f"\nExpert Return: {expert_return}")
    print(f"Custom IRL Return: {custom_return}")

if __name__ == "__main__":
    # Set up environment
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # Load expert demonstrations from precomputed data
    expert_demos = load_expert_data()

    # Compare Custom IRL with the precomputed expert
    compare_baselines(env, expert_demos)
