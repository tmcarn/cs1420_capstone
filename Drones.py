"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/Quadcopter-AI

This is where the players for the main game are defined
"""

import os

import pygame
from pygame.locals import *
import numpy as np
from math import pi


from PID.PID_controller import PID
import constants
from target import Target 

from stable_baselines3 import SAC, DQN
# from PPO_Env import PPOEnv


class Drone():
    
    def __init__(self, x, y) -> None:
        self.init_pos = np.array([x, y, 0]) # For respawning

        self.time_alive = 0
        self.target = None
        self.targets_hit = 0
        self.score = 0

        self.MAX_THRUST = 0.8
        self.MIN_TRUST = 0
     
        self.thrust = np.array([constants.THRUST_AMP, constants.THRUST_AMP])

        self.accs = np.zeros(3)
        self.vels = np.zeros(3)
        self.pos = np.array([x, y, 0])
        self.dt = 1/60

        self.im_pos = self.get_im_pos()

        self.dead = False

    def get_thrust(self, thrust, diff):
        thruster_left = constants.THRUST_AMP
        thruster_right = constants.THRUST_AMP

        thruster_left += thrust * constants.THRUST_AMP
        thruster_right += thrust * constants.THRUST_AMP
        
        thruster_left += diff * constants.DIFF_AMP
        thruster_right -= diff * constants.DIFF_AMP

        self.thrust = np.array([thruster_left, thruster_right])

        return self.thrust

    """
    Takes in thrust levels for both "propellers" and calculated new position based on rigid body equations
    """
    def update(self, new_thrust):
        self.thrust = new_thrust

        # Update Accelerations based on thrusts
        x_acc = - np.sum(self.thrust) * np.sin(self.pos[2] * (pi / 180))
        y_acc = - np.sum(self.thrust) * np.cos(self.pos[2] * (pi / 180)) + constants.GRAVITY
        theta_acc = (self.thrust[1] - self.thrust[0]) * constants.DRONE_LENGTH
        
        # Proogate down
        self.accs = np.array([x_acc, y_acc, theta_acc])
        self.vels += self.accs #* self.dt
        self.pos += self.vels # * self.dt

        self.im_pos = self.get_im_pos()

        return self.pos
    
    def reached_target(self, target):
        x_error, y_error = self.get_target_error(target)
        distance_to_target = np.linalg.norm(np.array([x_error, y_error]))

        if abs(distance_to_target) < constants.HIT_THRESH : # If its close enought to the target
            return True
        
        return False
    
    def get_target_error(self, target):
        pos_error = target.pos - self.pos[:-1]
        x_error = pos_error[0]
        y_error = pos_error[1]

        return x_error, y_error
    
    def get_observation(self, target):
        # Intrinsic
        angle_to_up = self.pos[2] / 180 * pi 
        mag_of_vel = np.linalg.norm(self.vels[:-1])
        angle_of_velocity = np.arctan2(self.vels[1], self.vels[0])
        
        # Relative to Target
        x_error, y_error = self.get_target_error(target)
         
        distance_to_target = np.linalg.norm(np.array([x_error, y_error]))
        angle_to_target = np.arctan2(y_error, x_error)
        angle_of_velocity = np.arctan2(self.vels[1], self.vels[0])
        angle_error = angle_to_target - angle_of_velocity

        return np.array([
            angle_to_up,
            mag_of_vel,
            angle_of_velocity,
            distance_to_target / 500, # Normalized
            angle_to_target,
            angle_error
        ], dtype='float32')

    def is_dead(self): # Checks if drone is beyond screen
        if (not (-constants.REC_DIST < self.pos[0] < constants.SCREEN_WIDTH + constants.REC_DIST) 
            or not (-constants.REC_DIST < self.pos[1] < constants.SCREEN_HEIGHT + constants.REC_DIST)):
            self.dead = True
            return True
        
        return False
    
    def respawn(self, x, y):
        self.__init__(x, y)
    
    def get_im_pos(self):
        self.im_pos = (self.pos[0] - constants.DRONE_WIDTH//2, self.pos[1] - constants.DRONE_WIDTH//2)
        return self.im_pos

class PIDDrone(Drone):
    def __init__(self, x, y):
        self.name = "PID"
        super().__init__(x,y)

        self.x_dist_PID = PID(0.2, 0, 0.2, -25, 25)
        self.angle_PID = PID(0.02, 0, 0.01, -1, 1)

        self.y_dist_PID = PID(2.5, 0, 1.5, -100, 100)
        self.y_vel_PID = PID(1, 0, 0, -1, 1)

    def act(self, target):

        x_error, y_error = self.get_target_error(target)

        x_vel = self.vels[0]
        y_vel = self.vels[1]
        ang_vel = self.vels[2]
        ang = self.pos[2]
        
        #PID Controllers
        ang_desired = self.x_dist_PID.compute(-x_error, self.dt)
        ang_error = ang_desired - ang
        diff_desired = self.angle_PID.compute(-ang_error, self.dt)

        y_vel_desired = self.y_dist_PID.compute(y_error, self.dt)
        y_vel_error = y_vel_desired - y_vel
        thrust_desired = self.y_vel_PID.compute(-y_vel_error, self.dt)

        # Update thrust levels
        self.thrust = self.get_thrust(thrust_desired, diff_desired)

        return self.thrust

    

class HumanDrone(Drone):
    def __init__(self, x, y):
        self.name = "PID"
        super().__init__(x,y)

    def act(self):
        pressed_keys = pygame.key.get_pressed()

        thruster_left = constants.THRUST_AMP
        thruster_right = constants.THRUST_AMP

        if pressed_keys[K_UP]:
            thruster_left += constants.THRUST_AMP
            thruster_right += constants.THRUST_AMP

        if pressed_keys[K_DOWN]:
            thruster_left -= constants.THRUST_AMP
            thruster_right -= constants.THRUST_AMP

        if pressed_keys[K_LEFT]:
            thruster_left -= constants.DIFF_AMP

        if pressed_keys[K_RIGHT]:
            thruster_right -= constants.DIFF_AMP

        self.thrust = np.array([thruster_left, thruster_right])

        return self.thrust

    
class SACDrone(Drone):
    def __init__(self, x, y, load_from=None):
        self.name = "SAC"
        super().__init__(x,y)
        
        if load_from != None:
            # Load Trained Model
            model_path = load_from
            self.model = SAC.load(model_path) 

    # For Testing
    def act(self):
        self.observation = self.get_observation(self.target)
        action, _states = self.model.predict(self.observation)

        (action0, action1) = (action[0], action[1])
        self.thrust = self.get_thrust(action0, action1)

        return self.thrust
        

class DQNDrone(Drone):
    def __init__(self, x, y, load_from=None):
        self.name = "DQN"
        super().__init__(x,y)
        self.DIFF_AMP = 0.0006 # More finite adjustment
        
        if load_from != None:
            # Load Trained Model
            model_path = load_from
            self.model = DQN.load(model_path) 

    def get_observation(self, target):
        # Intrinsic
        angle_to_up = self.pos[2] / 180 * pi 
        mag_of_vel = np.linalg.norm(self.vels[:-1])
        angle_of_velocity = np.arctan2(self.vels[1], self.vels[0])
        
        # Relative to Target
        x_error, y_error = self.get_target_error(target)
         
        distance_to_target = np.linalg.norm(np.array([x_error, y_error]))
        angle_to_target = np.arctan2(y_error, x_error)
        angle_of_velocity = np.arctan2(self.vels[1], self.vels[0])
        angle_error = angle_to_target - angle_of_velocity

        return np.array([
            angle_to_up,
            mag_of_vel,
            angle_of_velocity,
            distance_to_target / 500, # Normalized
            angle_to_target,
            angle_error
        ], dtype='float32')
    
    def get_thrust(self, action):
        thruster_left = constants.THRUST_AMP
        thruster_right = constants.THRUST_AMP

        if action == 0:
            pass
        elif action == 1:
            thruster_left += constants.THRUST_AMP
            thruster_right += constants.THRUST_AMP
        elif action == 2:
            thruster_left -= constants.THRUST_AMP
            thruster_right -= constants.THRUST_AMP
        elif action == 3:
            thruster_left += self.DIFF_AMP
            thruster_right -= self.DIFF_AMP
        elif action == 4:
            thruster_left -= self.DIFF_AMP
            thruster_right += self.DIFF_AMP

        return np.array([thruster_left, thruster_right])

    # For Testing
    def act(self):
        self.observation = self.get_observation(self.target)
        action, _states = self.model.predict(self.observation)

        (action0, action1) = (action[0], action[1])
        self.thrust = self.get_thrust(action0, action1)

        return self.thrust
        
