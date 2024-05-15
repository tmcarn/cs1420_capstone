
"""
Train a SAC agent using sb3 on the PIDRL_Env environment
"""

import os

from stable_baselines3 import SAC

from RL.PIDRL_Env import PIDRLEnv
from Drones import PIDSACDrone
import constants

model_num = "oscar_long2"

# Create models dir
models_dir = f"pid_models/SAC-{model_num}"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create log dir
logdir = "pid_logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create and wrap the environment
drone = PIDSACDrone(constants.SCREEN_WIDTH/2, constants.SCREEN_HEIGHT/2)
env = PIDRLEnv(drone)
env.reset()

# Create SAC agent
# model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda')
model = SAC.load("pid_models/SAC-oscar_long/150000.zip", env, device='cuda')

TIMESTEPS = 10000
iters = 0
for i in range(16,501):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"SAC-{model_num}")
    model.save(f"{models_dir}/{TIMESTEPS*(i)}")