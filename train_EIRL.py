import numpy as np
from huggingface_sb3 import load_from_hub
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

SEED = 42 # Does this matter?

def main():
    env = make_vec_env(
        "seals:seals/Ant-v1",
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals/Ant-v1",
        venv=env,
    )
    expert_rewards, _ = evaluate_policy(
        expert, env, 10, return_episode_rewards=True
    )
    print(np.mean(expert_rewards))

if __name__ == "__main__":
    main()
