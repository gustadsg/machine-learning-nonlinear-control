import os
import argparse
import time
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from sb3_contrib import ARS, TQC, TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

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
    def __init__(self, total_timesteps, algorithm_name, check_freq=5000):
        super(ProgressCallback, self).__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.timesteps_done = 0
        self.algorithm_name = algorithm_name
        self.start_time = time.time()

    def _on_step(self) -> bool:
        self.timesteps_done += 1
        if self.timesteps_done % self.check_freq == 0 or self.timesteps_done == self.total_timesteps:
            progress = 100 * self.timesteps_done / self.total_timesteps
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time * self.total_timesteps / self.timesteps_done
            remaining_time = estimated_total_time - elapsed_time
            estimated_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_time))
            print(f"[{self.algorithm_name}] Training progress: {progress:.2f}% -> {self.timesteps_done} out of {self.total_timesteps} timesteps.")
            print(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {remaining_time:.2f}s, Estimated end time: {estimated_end_time}")
        return True

def train_model(algorithm, train_timesteps):
    # create environment
    env = ElectricTapEnv()

    # train the model
    log_path = os.path.join("training", "logging", algorithm.lower())
    
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        learning_starts=10_000,
        verbose=1,
        tensorboard_log=log_path,
        # train_freq=(5, "episode"),
        # gradient_steps=50,
        action_noise=NormalActionNoise(mean=[0], sigma=[0.2]),
        # policy_kwargs={"net_arch": [400, 300]},
        # policy_delay=4,
        target_policy_noise=0.4,
        # gamma=0.99,
        # buffer_size=10_000_000
    )
    # model = TD3.load('./TD3.zip')
    model.set_env(env)

    callback = ProgressCallback(total_timesteps=train_timesteps, algorithm_name=algorithm)
    model.learn(total_timesteps=train_timesteps)

    # save the model
    model.save(algorithm)
    n_episodes = 20
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"Mean reward over {n_episodes} evaluation episodes for {algorithm}: {mean_reward}")

def main(train_timesteps):
    algorithms = [
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
