import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = RecordVideo(env, "./video/")
model = PPO.load("ppo-LunarLander-v2")

obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
