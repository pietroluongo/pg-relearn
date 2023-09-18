import gymnasium as gym
import panda_gym
from huggingface_sb3 import package_to_hub
from stable_baselines3.a2c import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env_id = "PandaReachDense-v3"
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
# We need to override the render_mode
eval_env.render_mode = "rgb_array"

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False


model = A2C.load(f"a2c-{env_id}")

package_to_hub(
    model=model,
    model_name=f"a2c-{env_id}",
    model_architecture="A2C",
    env_id=env_id,
    eval_env=eval_env,
    repo_id=f"pietroluongo/a2c-{env_id}",  # Change the username
    commit_message="Initial commit",
)
