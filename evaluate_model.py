from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy

from ElectricTapEnv import ElectricTapEnv

def run():
    model = TD3.load('./backup/TD3.zip')
    env = ElectricTapEnv(plot_results=True)
    n_episodes = 5
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"Mean reward over {n_episodes} evaluation episodes for TD3: {mean_reward}")
    

if __name__ == '__main__':
    run()