import os
import argparse
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from ElectricTapEnv import ElectricTapEnv

def main(episodes, algorithm, train_episodes):
    # create environment
    env = ElectricTapEnv()
    check_env(env)
    print("created env")

    # test environment
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, truncated, info = env.step(action)
            score += reward
            done = done
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

    # train the model
    log_path = os.path.join("training", "logs")
    
    if algorithm == 'PPO':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(log_path, "ppo"))
    elif algorithm == 'TD3':
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(log_path, "td3"))
    elif algorithm == 'SAC':
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(log_path, "sac"))
    else:
        raise ValueError("Unsupported algorithm. Choose from PPO, TD3, SAC.")
    
    model.learn(total_timesteps=train_episodes)

    # save the model
    model.save(algorithm)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward over 10 evaluation episodes: {mean_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a reinforcement learning model.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test the environment')
    parser.add_argument('--algorithm', type=str, default='SAC', help='Algorithm to use for training (PPO, TD3, SAC)')
    parser.add_argument('--train_timestaps', type=int, default=1_000_000, help='Number of episodes to train the model')
    
    args = parser.parse_args()
    
    main(args.episodes, args.algorithm, args.train_timestaps)
