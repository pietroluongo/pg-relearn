import gymnasium as gym
from huggingface_sb3 import package_to_hub
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

env_id = "LunarLander-v2"
model_architecture = "PPO"
model_name = "ppo-LunarLander-v2"

repo_id = f"pietroluongo/{model_name}"
commit_message = "Upload PPO LunarLander-v2 trained agent"

eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

model = PPO.load(model_name)

package_to_hub(
    model=model,  # Our trained model
    model_name=model_name,  # The name of our trained model
    model_architecture=model_architecture,  # The model architecture we used: in our case PPO
    env_id=env_id,  # Name of the environment
    eval_env=eval_env,  # Evaluation Environment
    repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
    commit_message=commit_message,
)
