import os
import argparse
from enum import Enum

from concurrent.futures import ThreadPoolExecutor
from stable_baselines3 import PPO, TD3, SAC, PPO, A2C, DDPG, HerReplayBuffer
from sb3_contrib import ARS, RecurrentPPO, TQC, TRPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from ElectricTapEnv import ElectricTapEnv

class Algorithm(Enum):
    PPO='PPO'
    TD3='TD3'
    SAC='SAC'
    A2C='A2C'
    DDPG='DDPG'
    ARS='ARS'
    TQC='TQC'
    TRPO='TRPO'

class ProgressCallback(BaseCallback):
    algorithm_color = {
        Algorithm.PPO.value: '\033[94m',  # Blue
        Algorithm.TD3.value: '\033[92m',  # Green
        Algorithm.SAC.value: '\033[93m',  # Yellow
        Algorithm.A2C.value: '\033[95m',  # Magenta
        Algorithm.DDPG.value: '\033[96m',  # Cyan
        Algorithm.ARS.value: '\033[90m',  # Grey
        Algorithm.TQC.value: '\033[35m',  # Purple
        Algorithm.TRPO.value: '\033[36m',  # Light Cyan
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
            color = self.algorithm_color[self.algorithm_name] if self.algorithm_name in self.algorithm_color else self.algorithm_color['RESET']
            print(f"{color}[{self.algorithm_name}] Training progress: {progress:.2f}% -> {self.timesteps_done} out of {self.total_timesteps} timesteps.")
        return True

def train_model(algorithm, train_timesteps):
    # create environment
    env = ElectricTapEnv()
    check_env(env)
    print(f"Created env for {algorithm}")

    # train the model
    log_path = os.path.join("training", "logs4", algorithm.lower())
    if algorithm == Algorithm.PPO.value:
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == Algorithm.TD3.value:
        model = TD3("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == Algorithm.SAC.value:
        model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == Algorithm.ARS.value:
        model = ARS("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == Algorithm.A2C.value:
        model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == Algorithm.DDPG.value:
        model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == Algorithm.TQC.value:
        model = TQC("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    elif algorithm == Algorithm.TRPO.value:
        model = TRPO("MlpPolicy", env, verbose=0, tensorboard_log=log_path)
    else:
        raise ValueError("Unsupported algorithm.")
    
    callback = ProgressCallback(total_timesteps=train_timesteps, algotithm_name=algorithm,check_freq=500)
    model.learn(total_timesteps=train_timesteps, callback=callback)

    # save the model
    model.save(algorithm)
    n_episodes = 20
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"Mean reward over {n_episodes} evaluation episodes for {algorithm}: {mean_reward}")

def main(train_timesteps):
    algorithms = [alg.value for alg in Algorithm]
    algorithms = [Algorithm.PPO.value]


    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_model, algorithm, train_timesteps) for algorithm in algorithms]
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train reinforcement learning models concurrently.')
    parser.add_argument('-t', type=int, default=100_000, help='Number of timesteps to train the model')
    
    args = parser.parse_args()
    
    main(args.t)
