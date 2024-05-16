import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pygame
import numpy as np
from math import pi
from random import randrange
from Drones import SACDrone, Drone
from target import Target
import constants


class RLEnv2(gym.Env):

    def __init__(self, drone:Drone):
        super().__init__()
        self.screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.time_alive_limit = 15

        self.past_thrusts = []

        # Load drone image
        self.drone_image = pygame.image.load(constants.DRONE_PATH)
        # Load target image
        self.target_image = pygame.image.load(constants.TARGET_PATH)

        # drone for training
        self.drone = drone
        
        init_target_pos = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
        self.target = Target(init_target_pos[0], init_target_pos[1])

        self.done = False


        # 2 action: thrust of each propeller
        self.action_space = spaces.Box(low=0, high=1, shape=(2,)) 

        # 9 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,))



    def step(self, action):
         # Game loop
        self.reward = 0.0

        (action0, action1) = (action[0], action[1])
        new_thrust = self.drone.get_thrust(action0, action1)

        for i in range(constants.UPDATES_PER_STEP): # Same action used for multiple steps, to speed up training

            self.drone.update(new_thrust)

            x_error, y_error = self.drone.get_target_error(self.target)
            distance_to_target = np.linalg.norm(np.array([x_error, y_error]))

            self.drone.time_alive += 1/60
            self.reward += 1/60 # Rewarded for Staying Alive
            self.reward -= distance_to_target / (60*100) # Staying alive more important than getting closer to target

            if len(self.past_thrusts)>0:
                prev_thust = self.past_thrusts[-1]
                thrust_dist = np.linalg.norm(prev_thust - new_thrust)
                self.reward -= thrust_dist * 100  # Penalty for inconsistent thrusts
            if distance_to_target > 1000:
                self.done = True
                self.reward -= 1000 # Penalized Heavily for Dying
                break # step finished

            elif self.drone.reached_target(self.target):
                self.done = False
                self.reward += 100 # Rewarded for Reaching Target
                self.drone.targets_hit += 1

                # Update Target
                new_target_position = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
                self.target = Target(new_target_position[0], new_target_position[1])


            elif self.drone.time_alive > self.time_alive_limit: # Exceeded Timer
                self.done = True
                break # step finished

            else:
                self.done = False

        self.past_thrusts.append(new_thrust)
        self.info = {}
        self.truncated = False
        self.observation = self.drone.get_observation(self.target)

        return self.observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.done = False
        drone_pos = np.random.uniform(0, constants.SCREEN_HEIGHT, (2,))
        self.drone.respawn(drone_pos[0], drone_pos[1])

        # Update Target
        new_target_position = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
        self.target = Target(new_target_position[0], new_target_position[1])

        self.observation = self.drone.get_observation(self.target)
        return self.observation, {}

    def render(self):
        self.screen.fill((200, 200, 200))
        # Display Target
        self.screen.blit(self.target_image, self.drone.target.im_pos)
        # Display Drone
        rotated_image = pygame.transform.rotate(self.drone_image, self.drone.pos[2])
        rotated_rect = rotated_image.get_rect(center=self.drone_image.get_rect(topleft=self.drone.im_pos).center)
        self.screen.blit(rotated_image, rotated_rect)

        # Update the display
        pygame.display.flip()
        # Cap the frame rate
        self.drone.dt = self.clock.tick(constants.FPS) / 1000

    # def close(self):
    #     ...
