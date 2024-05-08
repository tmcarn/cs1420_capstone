import pygame
import numpy as np
from random import randrange

from player import HumanDrone, PIDDrone
from target import Target
import constants
   
def main():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))

    clock = pygame.time.Clock()

    # Load drone image
    drone_image = pygame.image.load(constants.DRONE_PATH)

    # Load target image
    target_image = pygame.image.load(constants.TARGET_PATH)

    # Generate targets
    targets = []
    # num_targets = 1 if not constants.INTERACTIVE else constants.NUM_TARGETS
    for i in range(constants.NUM_TARGETS):
        targets.append(Target(randrange(0, constants.SCREEN_WIDTH), 
                              randrange(0, constants.SCREEN_WIDTH)))

    # Generate Drones
    # drones = [HumanDrone(screen.get_width()/2, screen.get_height()/2, targets), PIDDrone(screen.get_width()/2, screen.get_height()/2, targets)]
    drones = [PIDDrone(screen.get_width()/2, screen.get_height()/2, targets)]



    # Main loop
    running = True
    while running:
        screen.fill((200, 200, 200))

        # Quitable
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION and constants.INTERACTIVE:
                # Update target position with mouse movement
                target_x, target_y = event.pos
                targets[0].update_pos(target_x, target_y) # Only interact with one target

        for drone in drones:
            i = drone.target_num

            drone.current_target = targets[i]

            new_thrust = drone.act(targets[i]) 
            new_position = drone.update(new_thrust)

            if drone.is_dead():
                drone.respawn()

            elif drone.reached_target(targets[i]):
                if not constants.INTERACTIVE: drone.target_num += 1


            else:
                drone.time_alive += drone.dt

            # Display Target
            screen.blit(target_image, targets[i].im_pos)
            # Display Drone
            rotated_image = pygame.transform.rotate(drone_image, drone.pos[2])
            rotated_rect = rotated_image.get_rect(center=drone_image.get_rect(topleft=drone.im_pos).center)
            screen.blit(rotated_image, rotated_rect)

        # Update the display
        pygame.display.flip()
    
        # Cap the frame rate
        drone.dt = clock.tick(constants.FPS) / 1000

    # Quit Pygame
    pygame.quit()
     

if __name__=="__main__":
    # call the main function
    main()
