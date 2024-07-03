import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from ElectricTapEnv import ElectricTapEnv

class ProgressCallback(BaseCallback):
    algorithm_color = {
    'PPO': '\033[94m',  # Blue
    'TD3': '\033[92m',  # Green
    'SAC': '\033[93m',  # Yellow
    'RESET': '\033[0m'  # Reset color
}

    def __init__(self, total_timesteps, algotithm_name, check_freq=1000):
        super(ProgressCallback, self).__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.timesteps_done = 0
        self.algorithm_name = algotithm_name

    def _on_step(self) -> bool:
        self.timesteps_done += 1
        if self.timesteps_done % self.check_freq == 0 or self.timesteps_done == self.total_timesteps:
            progress = 100 * self.timesteps_done / self.total_timesteps
            print(f"{self.algorithm_color[self.algorithm_name]}[{self.algorithm_name}] Training progress: {progress:.2f}% -> {self.timesteps_done} out of {self.total_timesteps} timesteps.")
        return True

def train_model(algorithm, train_timesteps):
    # create environment
    env = ElectricTapEnv()
    check_env(env)
    print(f"Created env for {algorithm}")

    # train the model
    log_path = os.path.join("training", "logs2", algorithm.lower())

    if algorithm == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == 'TD3':
        model = TD3("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == 'SAC':
        model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    else:
        raise ValueError("Unsupported algorithm. Choose from PPO, TD3, SAC.")
    
    callback = ProgressCallback(total_timesteps=train_timesteps, algotithm_name=algorithm,check_freq=10_000)
    model.learn(total_timesteps=train_timesteps, callback=callback)

    # save the model
    model.save(algorithm)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward over 10 evaluation episodes for {algorithm}: {mean_reward}")

def main(episodes, train_timesteps):
    algorithms = ['PPO', 'TD3', 'SAC']

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_model, algorithm, train_timesteps) for algorithm in algorithms]
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train reinforcement learning models concurrently.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test the environment')
    parser.add_argument('--train_timesteps', type=int, default=100_000, help='Number of timesteps to train the model')
    
    args = parser.parse_args()
    
    main(args.episodes, args.train_timesteps)
