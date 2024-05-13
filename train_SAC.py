
"""
Train an PPO agent using sb3 on the PPO_ENV environment
"""

import os

from stable_baselines3 import SAC

from RL.RL_Env import RLEnv
from Drones import SACDrone
import constants

model_num = "3"

# Create models dir
models_dir = f"models/SAC-{model_num}"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create log dir
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create and wrap the environment
drone = SACDrone(constants.SCREEN_WIDTH/2, constants.SCREEN_HEIGHT/2)
env = RLEnv(drone)
env.reset()

# # Create SAC agent
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda')
# model = SAC.load("models/SAC-1_cont-cont/1900000.zip", env)

TIMESTEPS = 100000
iters = 0
for i in range(1,101):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"SAC-{model_num}")
    model.save(f"{models_dir}/{TIMESTEPS*(i)}")