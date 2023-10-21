import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


from ElectricTapEnv import ElectricTapEnv

# create environment
env = ElectricTapEnv()
check_env(env)

env = DummyVecEnv([lambda: env])

# test environment
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

# train the model
log_path = os.join("training", "logs")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

num_of_episodes = 1000000
model.learn(total_timesteps=num_of_episodes)

# save the model
model.save('PPO')
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward over 10 evaluation episodes: {mean_reward}")