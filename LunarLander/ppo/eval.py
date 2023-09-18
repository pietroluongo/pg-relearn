import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("LunarLander-v2"))

model = PPO.load("ppo-LunarLander-v2")

mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
