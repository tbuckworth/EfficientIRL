import numpy as np
from huggingface_sb3 import load_from_hub
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import eirl

SEED = 42 # Does this matter?

def main():
    training_increments = 5
    n_epochs = 1000
    default_rng = np.random.default_rng(SEED)
    env = make_vec_env(
        "seals:seals/Ant-v1",
        rng=default_rng,
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
    target_rewards = np.mean(expert_rewards)
    print(target_rewards)

    expert_rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=default_rng,
    )
    expert_transitions = rollout.flatten_trajectories(expert_rollouts)

    expert_trainer = eirl.EIRL(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_transitions,
        rng=default_rng,
    )
    for i, increment in enumerate([training_increments for i in range(n_epochs // training_increments)]):
        expert_trainer.train(n_epochs=increment)
        rewards, _ = evaluate_policy(
            expert_trainer.policy, env, 10, return_episode_rewards=True
        )
        mean_rew = np.mean(rewards)
        std_err = np.std(rewards) / np.sqrt(len(rewards))
        per_expert = mean_rew/target_rewards
        print(f"Epoch:{(i+1)*increment}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")



if __name__ == "__main__":
    main()
