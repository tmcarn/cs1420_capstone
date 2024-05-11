from stable_baselines3.common.env_checker import check_env

from RL.RL_Env import RLEnv
from Drones import SACDrone

drone = SACDrone(0.0,0.0)

env = RLEnv(drone)

check_env(env)