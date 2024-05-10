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

from stable_baselines3 import PPO, SAC
# from PPO_Env import PPOEnv


class Drone():
    
    def __init__(self, x, y) -> None:
        self.init_pos = np.array([x, y, 0]) # For respawning
        
        self.x_error = None
        self.y_error = None
        self.distance_to_target = None
        self.angle_to_target = None
        self.angle_of_velocity = None
        # Angle between the to_target vector and the velocity vector
        self.angle_error = None

        self.current_target = None
        self.time_alive = 0
        self.target_num = 0
        self.score = 0

        self.MAX_THRUST = 0.8
        self.MIN_TRUST = 0
     
        self.thrust = np.array([constants.THRUST_AMP, constants.THRUST_AMP])

        self.accs = np.zeros(3)
        self.vels = np.zeros(3)
        self.pos = np.array([x, y, 0])
        self.dt = constants.DT

        self.im_pos = self.get_im_pos()

        self.dead = False

        

    def update(self, new_thrust):
        self.thrust = new_thrust

        # Update Accelerations based on thrusts
        x_acc = - np.sum(self.thrust) * np.sin(self.pos[2] * (pi / 180))
        y_acc = - np.sum(self.thrust) * np.cos(self.pos[2] * (pi / 180)) + constants.GRAVITY
        theta_acc = (self.thrust[1] - self.thrust[0]) * constants.DRONE_LENGTH
        
        # Proogate down
        self.accs = np.array([x_acc, y_acc, theta_acc])
        self.vels += self.accs * constants.DT
        self.pos += self.vels * constants.DT

        self.im_pos = self.get_im_pos()

        return self.pos
    
    def reached_target(self):
        self.get_target_error()

        if abs(self.distance_to_target) < constants.HIT_THRESH :
            return True
        
        return False
    
    def get_target_error(self):
        pos_error = self.current_target.pos - self.pos[:-1]

        self.x_error = pos_error[0]
        self.y_error = pos_error[1]
        self.distance_to_target = np.linalg.norm(np.array([self.x_error, self.y_error]))
        
        self.angle_to_target = np.arctan2(self.y_error, self.x_error)

        self.angle_of_velocity = np.arctan2(self.vels[1], self.vels[0])
        # Angle between the to_target vector and the velocity vector
        self.angle_error = self.angle_to_target - self.angle_of_velocity

        return self.distance_to_target / 500, self.angle_to_target, self.angle_error
    
    def assign_new_target(self, x, y):
        self.current_target = Target(x,y)

    
    def is_dead(self):
        if (not (-constants.REC_DIST < self.pos[0] < constants.SCREEN_WIDTH + constants.REC_DIST) 
            or not (-constants.REC_DIST < self.pos[1] < constants.SCREEN_HEIGHT + constants.REC_DIST)):
            self.dead = True
            return True
        
        return False
    
    def respawn(self):
        self.__init__(self.init_pos[0], self.init_pos[1])

    
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

    def act(self):
        thruster_left = constants.THRUST_AMP
        thruster_right = constants.THRUST_AMP

        self.get_target_error()

        self.x_error
        self.y_error
        x_vel = self.vels[0]
        y_vel = self.vels[1]
        ang_vel = self.vels[2]
        ang = self.pos[2]
        
        #PID Controllers
        ang_desired = self.x_dist_PID.compute(-self.x_error, self.dt)
        ang_error = ang_desired - ang
        diff_desired = self.angle_PID.compute(-ang_error, self.dt)

        y_vel_desired = self.y_dist_PID.compute(self.y_error, self.dt)
        y_vel_error = y_vel_desired - y_vel
        thrust_desired = self.y_vel_PID.compute(-y_vel_error, self.dt)

        thruster_left += thrust_desired * constants.THRUST_AMP
        thruster_right += thrust_desired * constants.THRUST_AMP
        
        thruster_left += diff_desired * constants.DIFF_AMP
        thruster_right -= diff_desired * constants.DIFF_AMP

        # Update thrust levels
        self.thrust = np.array([thruster_left, thruster_right])

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

    
class PPODrone(Drone):
    def __init__(self, x, y):
        self.name = "PPO"
        super().__init__(x,y)

        # Create and wrap the environment
        # env = PPOEnv()
        # env.reset()

        model_path = "models/PPO/290000.zip"

        self.model = PPO.load(model_path) 

    def get_observation(self):
        # Intrinsic
        angle_to_up = self.pos[2] / 180 * pi 
        mag_of_vel = np.linalg.norm(self.vels[:-1])
        angle_of_velocity = np.arctan2(self.vels[1], self.vels[0])
        
        # Relative to Target
        distance_to_target, angle_to_target, angle_error = self.get_target_error()

        return np.array([
            angle_to_up,
            mag_of_vel,
            angle_of_velocity,
            distance_to_target,
            angle_to_target,
            angle_error
        ], dtype='float32')

    def act(self):
        self.observation = self.get_observation()
        action, _states = self.model.predict(self.observation)

        (action0, action1) = (action[0], action[1])

        thruster_left = constants.THRUST_AMP
        thruster_right = constants.THRUST_AMP

        thruster_left += action0 * constants.THRUST_AMP
        thruster_right += action0 * constants.THRUST_AMP
        thruster_left += action1 * constants.DIFF_AMP
        thruster_right -= action1 * constants.DIFF_AMP

        self.thrust = np.array([thruster_left, thruster_right])

        return self.thrust



class SACDrone(Drone):
    def __init__(self, x, y):
        self.name = "PPO"
        super().__init__(x,y)

        # Create and wrap the environment
        # env = PPOEnv()
        # env.reset()

        model_path = "models/SAC-1/20000.zip"

        self.model = SAC.load(model_path) 

    def get_observation(self):
        # Intrinsic
        angle_to_up = self.pos[2] / 180 * pi 
        mag_of_vel = np.linalg.norm(self.vels[:-1])
        angle_of_velocity = np.arctan2(self.vels[1], self.vels[0])
        
        # Relative to Target
        distance_to_target, angle_to_target, angle_error = self.get_target_error()

        return np.array([
            angle_to_up,
            mag_of_vel,
            angle_of_velocity,
            distance_to_target,
            angle_to_target,
            angle_error
        ], dtype='float32')

    def act(self):
        self.observation = self.get_observation()
        action, _states = self.model.predict(self.observation)

        (action0, action1) = (action[0], action[1])

        thruster_left = constants.THRUST_AMP
        thruster_right = constants.THRUST_AMP

        thruster_left += action0 * constants.THRUST_AMP
        thruster_right += action0 * constants.THRUST_AMP
        thruster_left += action1 * constants.DIFF_AMP
        thruster_right -= action1 * constants.DIFF_AMP

        self.thrust = np.array([thruster_left, thruster_right])

        return self.thrust
