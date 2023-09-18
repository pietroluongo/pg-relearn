import gym
import gym_pygame

from torch.optim import Adam
from policy import Policy
import utils
import pickle

env_id = "Pixelcopter-PLE-v0"
env = gym.make(env_id)
eval_env = gym.make(env_id)

s_size, a_size = utils.get_env_dimensions(env)

pixelcopter_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 50000,
    # "n_training_episodes": 5,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

pixelcopter_policy = Policy(
    state_sz=pixelcopter_hyperparameters["state_space"],
    action_sz=pixelcopter_hyperparameters["action_space"],
    hidden_sz=pixelcopter_hyperparameters["h_size"],
)

pixelcopter_optimizer = Adam(
    pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"]
)


scores = pixelcopter_policy.reinforce(
    optimizer=pixelcopter_optimizer,
    n_training_episodes=pixelcopter_hyperparameters["n_training_episodes"],
    max_t=pixelcopter_hyperparameters["max_t"],
    gamma=pixelcopter_hyperparameters["gamma"],
    print_every=100,
    env=env,
)

mean_reward, std_reward = pixelcopter_policy.eval(
    eval_env,
    pixelcopter_hyperparameters["max_t"],
    pixelcopter_hyperparameters["n_evaluation_episodes"],
)

print(f"Result: {mean_reward:.2f} +- {std_reward:.2f}")

dat = {
    # "env": env,
    "model": pixelcopter_policy,
    "params": pixelcopter_hyperparameters,
    "optimizer": pixelcopter_optimizer,
}

with open("stuff.dat", "wb") as f:
    pickle.dump(dat, f)
    f.close()
