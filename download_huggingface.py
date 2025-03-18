import numpy as np
from imitation.data import rollout
from stable_baselines3 import TD3

from huggingface_sb3 import load_from_hub
from stable_baselines3.common.evaluation import evaluate_policy

from helper_local import load_env



def main():
    # Load the TD3 checkpoint from Hugging Face
    td3_checkpoint = load_from_hub(
        repo_id="qgallouedec/td3-Hopper-v3-3855171845",
        filename="td3-Hopper-v3.zip",
    )

    # You may now need to convert this checkpoint into an imitation-compatible policy.
    # This could involve loading the model with Stable Baselines3 and then wrapping it:
    model = TD3.load(td3_checkpoint)

    # Wrap or adapt 'model' as needed for your imitation training routines.
    expert = model.policy
    env_name = "seals/Hopper-v1"
    n_envs = 2
    seed = 0
    n_eval_episodes = 10
    default_rng, env = load_env(env_name, n_envs, seed)

    expert_rewards, _ = evaluate_policy(
        expert, env, n_eval_episodes, return_episode_rewards=True
    )
    target_rewards = np.mean(expert_rewards)
    print(f"Target:{target_rewards}")

    expert_rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=5),
        rng=default_rng,
        exclude_infos=False,
    )
    expert_transitions = rollout.flatten_trajectories_with_rew(expert_rollouts)


if __name__ == "__main__":
    main()
