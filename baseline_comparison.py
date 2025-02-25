from imitation.algorithms import bc, gail
from imitation.data import rollout
from stable_baselines3 import PPO

def compare_baselines(env, expert_demos):
    print("\nTraining Behavioral Cloning (BC)...")
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_demos
    )
    bc_trainer.train(n_epochs=5)
    bc_return = rollout.rollout_stats(rollout.generate_trajectories(bc_trainer.policy, env, 10))['return_mean']

    print("\nTraining GAIL...")
    gail_trainer = gail.GAIL(
        demonstrations=expert_demos,
        venv=env,
        gen_algo=PPO("MlpPolicy", env, verbose=1)
    )
    gail_trainer.train(total_timesteps=50000)
    gail_return = rollout.rollout_stats(rollout.generate_trajectories(gail_trainer.policy, env, 10))['return_mean']

    print(f"BC Return: {bc_return}, GAIL Return: {gail_return}")

if __name__ == "__main__":
    from custom_irl import env, expert_demos
    compare_baselines(env, expert_demos)
