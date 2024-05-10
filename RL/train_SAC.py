
"""
Train an PPO agent using sb3 on the PPO_ENV environment
"""

import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


from RL.RL_Env import RLEnv

model_num = 50


# Create models dir
models_dir = f"models/SAC-{model_num}"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create log dir
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create and wrap the environment
env = RLEnv()
env.reset()

# Create PPO agent
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"SAC-{model_num}")
    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")