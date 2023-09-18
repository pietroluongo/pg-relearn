from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import A2C
import panda_gym
from numpy.random import randint

env_id = "PandaPickAndPlace-v3"
env = make_vec_env(env_id, n_envs=16, seed=randint(0, 2**16 - 1))


env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


model = A2C(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(1_000_000, progress_bar=True)
model.save(f"a2c-{env_id}")
env.save("vec_normalize.pkl")
