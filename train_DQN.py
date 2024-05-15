
"""
Train an DQN agent using sb3 on the RL_Env environment
"""

import os

from stable_baselines3 import DQN

from RL.RL_Env import RLEnv
from Drones import DQNDrone
import constants

model_num = "1"

# Create models dir
models_dir = f"models/DQN-{model_num}"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create log dir
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create and wrap the environment
drone = DQNDrone(constants.SCREEN_WIDTH/2, constants.SCREEN_HEIGHT/2)
env = RLEnv(drone)
env.reset()

# # Create SAC agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

# model = SAC.load("models/SAC-1_continued/1600000.zip", env)

TIMESTEPS = 100000
iters = 0
for i in range(1,101):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN-{model_num}")
    model.save(f"{models_dir}/{TIMESTEPS*(i)}")