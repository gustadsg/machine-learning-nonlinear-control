from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy

from ElectricTapEnv import ElectricTapEnv
from ReducedEnv import ReducedEnv

def run():
    model = TD3.load('./TD3.zip')
    env = ElectricTapEnv(plot_results=True)
    # env = ReducedEnv(plot_results=True) # usado para modelo reduzido treinado no google colab
    n_episodes = 40
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"Mean reward over {n_episodes} evaluation episodes for TD3: {mean_reward}")
    

if __name__ == '__main__':
    run()