import gymnasium as gym
import panda_gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.a2c import A2C
from sb3_contrib.common.wrappers.time_feature import TimeFeatureWrapper
import numpy as np
import time

env_id = "PandaReachDense-v3"
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3", render_mode="human")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
model = A2C.load(f"a2c-{env_id}")

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False


obs = eval_env.reset()

while True:
    action, _ = model.predict(observation=obs)
    eval_env.step_async(action)
    time.sleep(0.3333333333)
    obs, reward, done, info = eval_env.step_wait()
    if done:
        time.sleep(1)
        obs = eval_env.reset()

env.close()
