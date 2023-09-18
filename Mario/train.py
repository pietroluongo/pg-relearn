import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
import numpy as np
import flappy_bird_env
import matplotlib.pyplot as plt

model_name = "ppo-Flappy-v0"


def debug_env(env):
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", env.observation_space.shape)
    print(
        "Sample observation", env.observation_space.sample()
    )  # Get a random observation


print("Starting env...")


# env = make_vec_env(
#     "FlappyBird-v0", 32, seed=np.random.randint(0, 2**31 - 1), wrapper_class=WarpFrame
# )


eenv = WarpFrame(gym.make("FlappyBird-v0", render_mode="rgb_array"))
eenv.reset()
# eenv.render()
while True:
    action = eenv.action_space.sample()
    obs, reward, terminated, _, info = eenv.step(action)
    eenv.render()

    print(obs.shape)
    print(obs)
    # Checking if the player is still alive
    plt.imshow(obs)
    plt.show()
    if terminated:
        break
eenv.close()


# plt.imshow(obs)
exit(1)

print("Creating model...")
model = PPO(
    policy="CnnPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorlogs",
)

# print(f"Loading model {model_name}")
# model.load(f"./{model_name}")
# print(f"Model loaded.")
model.learn(total_timesteps=1000000, progress_bar=True)
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
