import pygame
import numpy as np

from RL.RL_Env import RLEnv

from Drones import HumanDrone, PIDDrone, RLDrone, SACDrone
import constants
   
def main():

    drone = SACDrone(constants.SCREEN_WIDTH/2, constants.SCREEN_HEIGHT/2, testing=True)
    env = RLEnv(drone)

    # Main loop
    running = True
    while running:
        env.act()
        
        env.render()

    # Quit Pygame
    pygame.quit()
     

if __name__=="__main__":
    # call the main function
    main()
