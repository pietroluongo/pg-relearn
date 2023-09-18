import torch
from gymnasium.core import Env
from typing import Any


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_env_dimensions(env) -> tuple[int, int]:
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    return state_size, action_size


def debug_env(env):
    debug_env_obs_space(env)
    debug_env_action_space(env)


def debug_env_obs_space(env):
    s_size, _ = get_env_dimensions(env)
    print(f"==========OBSERVATION SPACE==========")
    print("The State Space is: ", s_size)
    print(f"Sample observation: {env.observation_space.sample()}")
    print(f"=====================================")


def debug_env_action_space(env):
    _, a_size = get_env_dimensions(env)
    print(f"============ACTION  SPACE============")
    print("The Action Space is: ", a_size)
    print(f"Action Space Sample: {env.action_space.sample()}")
    print(f"=====================================")
