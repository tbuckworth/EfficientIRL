import numpy as np
import torch
from huggingface_sb3 import load_from_hub
from imitation.data import rollout, wrappers
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards import reward_wrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import eirl
from ant_v1_learner_config import load_ant_learner

SEED = 42  # Does this matter?


def wrap_env_with_reward(env, policy):
    def predict_processed(
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            done: np.ndarray,
            **kwargs,
    ) -> np.ndarray:
        # this is for the reward function signature
        with torch.no_grad():
            nobs = torch.FloatTensor(next_state).to(device=policy.device)
        return policy.predict_values(nobs).squeeze().detach().cpu().numpy()


    venv_buffering = wrappers.BufferingWrapper(env)
    venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
        venv_buffering,
        reward_fn=predict_processed,
    )
    return venv_wrapped


def main():
    training_increments = 5
    n_epochs = 20
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
    learner = load_ant_learner(wrap_env_with_reward(env, expert_trainer.policy))
    learner.learn(10_000)
    for i, increment in enumerate([training_increments for i in range(n_epochs // training_increments)]):
        expert_trainer.train(n_epochs=increment,progress_bar=False)
        mean_rew, per_expert, std_err = evaluate(env, expert_trainer, target_rewards)
        print(f"Epoch:{(i + 1) * increment}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")

    learner = load_ant_learner(wrap_env_with_reward(env, expert_trainer.policy))
    for i in range(20):
        learner.learn(10_000)
        mean_rew, per_expert, std_err = evaluate(env, expert_trainer, target_rewards)
        print(f"Timesteps:{(i + 1) * 10_000}\tMeanRewards:{mean_rew:.1f}\tStdError:{std_err:.2f}\tRatio{per_expert:.2f}")


def evaluate(env, expert_trainer, target_rewards):
    rewards, _ = evaluate_policy(
        expert_trainer.policy, env, 10, return_episode_rewards=True
    )
    mean_rew = np.mean(rewards)
    std_err = np.std(rewards) / np.sqrt(len(rewards))
    per_expert = mean_rew / target_rewards
    return mean_rew, per_expert, std_err


if __name__ == "__main__":
    main()
