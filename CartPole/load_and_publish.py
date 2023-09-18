import gymnasium as gym
from torch.optim import Adam
from policy import Policy
from publish import push_to_hub
import utils
import pickle

env_id = "CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")
eval_env = gym.make(env_id)

dat = {}

with open("stuff.dat", "rb") as f:
    dat = pickle.load(f)
    f.close()

repo_id = f"pietroluongo/Reinforce-{env_id}"  # TODO Define your repo id {username/Reinforce-{model-id}}
print("Publishing...")
push_to_hub(
    repo_id,
    dat["model"],  # The model we want to save
    dat["params"],  # Hyperparameters
    env,  # Video environment
    eval_env,  # Evaluation environment
    env_id,
    video_fps=30,
)
