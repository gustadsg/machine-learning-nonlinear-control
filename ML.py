import os
import argparse
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from sb3_contrib import ARS, TQC, TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from ElectricTapEnv import ElectricTapEnv

class Algorithm(Enum):
    PPO = 'PPO'
    TD3 = 'TD3'
    SAC = 'SAC'
    A2C = 'A2C'
    DDPG = 'DDPG'
    ARS = 'ARS'
    TQC = 'TQC'
    TRPO = 'TRPO'

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

    def __init__(self, total_timesteps, algorithm_name, check_freq=1000):
        super(ProgressCallback, self).__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.timesteps_done = 0
        self.algorithm_name = algorithm_name

    def _on_step(self) -> bool:
        self.timesteps_done += 1
        if self.timesteps_done % self.check_freq == 0 or self.timesteps_done == self.total_timesteps:
            progress = 100 * self.timesteps_done / self.total_timesteps
            color = self.algorithm_color.get(self.algorithm_name, self.algorithm_color['RESET'])
            print(f"{color}[{self.algorithm_name}] Training progress: {progress:.2f}% -> {self.timesteps_done} out of {self.total_timesteps} timesteps.")
        return True

def train_model(algorithm, train_timesteps):
    # create environment
    env = ElectricTapEnv()
    # check_env(env)  # You can skip this if confident about env's correctness
    print(f"Created env for {algorithm}")

    # train the model
    log_path = os.path.join("training", "logs5", algorithm.lower())
    model_cls = {
        Algorithm.PPO.value: PPO,
        Algorithm.TD3.value: TD3,
        Algorithm.SAC.value: SAC,
        Algorithm.A2C.value: A2C,
        Algorithm.DDPG.value: DDPG,
        Algorithm.ARS.value: ARS,
        Algorithm.TQC.value: TQC,
        Algorithm.TRPO.value: TRPO,
    }[algorithm]
    
    model = model_cls("MlpPolicy", env, verbose=0, tensorboard_log=log_path, )

    callback = ProgressCallback(total_timesteps=train_timesteps, algorithm_name=algorithm, check_freq=500)
    model.learn(total_timesteps=train_timesteps, callback=callback)

    # save the model
    model.save(algorithm)
    n_episodes = 20
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"Mean reward over {n_episodes} evaluation episodes for {algorithm}: {mean_reward}")

def main(train_timesteps):
    algorithms = [
        Algorithm.SAC.value,
        Algorithm.DDPG.value,
        Algorithm.A2C.value,
        Algorithm.ARS.value,
        Algorithm.TD3.value
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(train_model, algorithm, train_timesteps) for algorithm in algorithms]
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train reinforcement learning models concurrently.')
    parser.add_argument('-t', type=int, default=100_000, help='Number of timesteps to train the model')
    
    args = parser.parse_args()
    
    main(args.t)
