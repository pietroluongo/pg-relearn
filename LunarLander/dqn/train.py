import gymnasium as gym

# from huggingface_sb3 import load_from_hub, package_to_hub
# from huggingface_hub import notebook_login

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def debug_env(env):
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", env.observation_space.shape)
    print(
        "Sample observation", env.observation_space.sample()
    )  # Get a random observation


print("Starting env...")

env = make_vec_env("LunarLander-v2", 16)

print("Creating model...")
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=1000000)
model_name = "ppo-LunarLander-v2"
model.save(model_name)
# obs, info = env.reset()

# debug_env(env)

# for _ in range(20):
#     action = env.action_space.sample()
#     print(f"Taken action: {action}")
#     obs, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         print("Reset env")
#         obs, info = env.reset()

# env.close()
