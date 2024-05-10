import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pygame
import numpy as np
from math import pi
from player import Drone
import constants


class RLEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # Set up the display
        # self.screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.time_alive_limit = 50

        # Load drone image
        self.drone_image = pygame.image.load(constants.DRONE_PATH)
        # Load target image
        self.target_image = pygame.image.load(constants.TARGET_PATH)

        
        self.drone = Drone(constants.SCREEN_WIDTH/2, constants.SCREEN_HEIGHT/2)
        
        self.target_position = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
        self.drone.assign_new_target(self.target_position[0], self.target_position[1])

        self.done = False


        # 2 action thrust amplitude and thrust difference in float values between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # 6 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

    def step(self, action):
         # Game loop
        self.reward = 0.0
        (action0, action1) = (action[0], action[1])

        thruster_left = constants.THRUST_AMP
        thruster_right = constants.THRUST_AMP

        thruster_left += action0 * constants.THRUST_AMP
        thruster_right += action0 * constants.THRUST_AMP
        thruster_left += action1 * constants.DIFF_AMP
        thruster_right -= action1 * constants.DIFF_AMP

        new_thrust = np.array([thruster_left, thruster_right])

        self.drone.update(new_thrust)
        self.observation = self.get_observation()

        self.drone.time_alive += self.drone.dt
    
        self.reward += self.drone.dt / 60 # Rewarded for Staying Alive
        self.reward -= self.drone.distance_to_target / (60*100) # Penalized for being far from target

        # self.reward -= self.time_since_target / 120 # Penalized for being slow

        if self.drone.is_dead() or self.drone.distance_to_target > 1000:
            self.done = True
            self.drone.time_alive = 0
            self.reward -= 1000 # Penalized Heavily for Dying

        
        elif self.drone.reached_target():
            self.done = False
            self.reward += 100 # Rewarded for Reaching Target
            self.drone.target_num += 1

            self.target_position = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
            self.drone.assign_new_target(self.target_position[0], self.target_position[1])

        # elif self.time_since_target > self.time_since_target_limit:
        #     self.done = True

        elif self.drone.time_alive > self.time_alive_limit: 
            self.done = True

        self.info = {}
        self.truncated = False
        return self.observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.done = False
        self.drone.respawn()

        self.target_position = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
        self.drone.assign_new_target(self.target_position[0], self.target_position[1])

        self.observation = self.get_observation()
        return self.observation, {}
    
    def get_observation(self):
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
        ], dtype='float32')


    # def render(self):
    #     ...

    # def close(self):
    #     ...
