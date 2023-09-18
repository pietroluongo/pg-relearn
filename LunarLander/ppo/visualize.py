import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="human")

model = PPO.load("ppo-LunarLander-v2")

obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
