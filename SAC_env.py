import gym
import numpy as np
from math import pi

from player import Drone
import constants 

import pygame
from pygame.locals import *


class SAC_Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.drone = None

        # 2 action thrust amplitude and thrust difference in float values between -1 and 1
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        # 6 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=float)
    
    def reset(self):
        self.__init__()

    def get_obs(self) -> np.ndarray:

        # Intrinsic
        angle_to_up = self.drone.pos[2] / 180 * pi 
        mag_of_vel = np.linalg.norm(self.drone.vels[:-1])
        angle_of_velocity = np.arctan2(self.drone.vels[1], self.drone.vels[0])
        
        # Relative to Target
        distance_to_target, angle_to_target, angle_error = self.drone.get_target_error()

        return np.array([
            angle_to_up,
            mag_of_vel,
            angle_of_velocity,
            distance_to_target,
            angle_to_target,
            angle_error
        ])

    def step(self, action):
        self.drone = Drone(0,0) 
        render = False
        self.drone.reward = 0.0
        (action0, action1) = (action[0], action[1])

        for i in range(1):

            thruster_left = constants.THRUST_AMP
            thruster_right = constants.THRUST_AMP

            thruster_left += action0 * constants.THRUST_AMP
            thruster_right += action0 * constants.THRUST_AMP
            thruster_left += action1 * constants.DIFF_AMP
            thruster_right -= action1 * constants.DIFF_AMP

            new_thrust = np.array([thruster_left, thruster_right])

            # Update
            self.drone.update(new_thrust)

            # Get Errors from Target
            self.drone.get_target_error()

            # Reward per step survived
            self.drone.score += 1 / 60
            # Penalty according to the distance to target
            self.drone.score -= self.drone.distance_to_target / (100 * 60)

            if self.drone.reached_target():
                self.drone.score += 100

            # If out of time
            if self.drone.time_alive > 20:
                done = True
                break

            # If too far from target (crash)
            elif abs(self.drone.distance_to_target) > 1000:
                self.drone.score -= 1000
                done = True
                break

            else:
                done = False

            if render:
                self.render()


    def render(self, mode):
        pass

    def close(self):
        pass