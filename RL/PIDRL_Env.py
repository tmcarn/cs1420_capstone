import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pygame
import numpy as np
from math import pi
from Drones import PIDSACDrone, PIDDrone
from target import Target
import constants


class PIDRLEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, drone, baseline):
        super().__init__()
        self.screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.time_alive_limit = 10
        self.time_since_hit = 0

        # Load drone image
        self.drone_image = pygame.image.load(constants.DRONE_PATH)
        # Load target image
        self.target_image = pygame.image.load(constants.TARGET_PATH)

        # drone for training
        self.drone = drone
        self.baseline = baseline

        self.drones = [baseline, drone]
        
        # Random target position
        init_target_pos = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
        self.target = Target(init_target_pos[0], init_target_pos[1])

        self.done = False

        # 4 PID controllers, 4 parameters per controller
        self.action_space = spaces.Box(low=-5, high=5, shape=(4,4)) 
        # 6 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))



    def step(self, action):
         # Game loop
        self.reward = 0.0
        self.time_since_hit = 0

        # Update PIDs based on action
        self.drone.x_dist_PID.tune(action[0][0], action[0][1], action[0][2], action[0][3])
        self.drone.angle_PID.tune(action[1][0], action[1][1], action[1][2], action[1][3])

        self.drone.y_dist_PID.tune(action[2][0], action[2][1], action[2][2], action[2][3])
        self.drone.y_vel_PID.tune(action[3][0], action[3][1], action[3][2], action[3][3])


        # while not self.done: # Same pid for entire run
        for i in range(constants.UPDATES_PER_STEP): # Step 10 actions with updated PIDs


            # Get new thrust based on PID and Target
            new_thrust = self.drone.pid_compute(self.target)

            self.drone.update(new_thrust) # updates position based on new thrust
            # self.observation = self.drone.get_observation(self.target) # gets new obs based on updated position

            x_error, y_error = self.drone.get_target_error(self.target)
            distance_to_target = np.linalg.norm(np.array([x_error, y_error]))

            self.drone.time_alive += 1/60
            self.time_since_hit += 1/60
            self.reward += 1/60 # Rewarded for Staying Alive
            self.reward -= distance_to_target / (60*100) # Staying alive more important than getting closer to target

            if distance_to_target > 1000: # Out of range
                self.done = True
                self.reward -= 1000 # Penalized Heavily for Dying
                break # step finished

            elif self.drone.reached_target(self.target):
                self.done = False
                self.reward += 100 # Rewarded for Reaching Target
                # self.reward += 100.0/self.time_since_hit # Incentive for being quicker
                self.time_since_hit = 0
                self.drone.targets_hit += 1

                # Update Target
                new_target_position = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
                self.target = Target(new_target_position[0], new_target_position[1])


            elif self.drone.time_alive > self.time_alive_limit: # Exceeded Timer
                self.done = True
                break # step finished

        self.info = {}
        self.truncated = False
        self.observation = self.drone.get_observation(self.target) # gets new obs based on updated position


        return self.observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.done = False
        
        # Update Drone Starting Position
        drone_pos = np.random.uniform(0, constants.SCREEN_HEIGHT, (2,))
        self.drone.respawn(drone_pos[0], drone_pos[1]) #Resets to intial PID tune

        # Update Target Position
        new_target_position = np.random.randint(constants.SCREEN_HEIGHT, size=(2,))
        self.target = Target(new_target_position[0], new_target_position[1])

        # Get new initial observation
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
