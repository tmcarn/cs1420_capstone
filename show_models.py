import pygame
import numpy as np


from Drones import HumanDrone, PIDDrone, SACDrone
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
    for i in range(constants.NUM_TARGETS):
        target_positions = np.random.randint(constants.SCREEN_HEIGHT, size=(constants.NUM_TARGETS, 2))

    drones = [SACDrone(screen.get_width()/2, screen.get_height()/2, load_from='models/SAC-1_cont-cont/1800000.zip')]

    for drone in drones:
        # Update Target
        drone.target = Target(target_positions[0][0], target_positions[0][1])

    # Main loop
    running = True
    while running:
        screen.fill((200, 200, 200))

        # # Quitable
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     elif event.type == pygame.MOUSEMOTION and constants.INTERACTIVE:
        #         # Update target position with mouse movement
        #         target_x, target_y = event.pos
        #         drone.target = Target(target_x, target_y) # Only interact with one target

        for drone in drones:

            new_thrust = drone.act() 
            drone.update(new_thrust)

            if drone.is_dead():
                drone.respawn(screen.get_width()/2, screen.get_height()/2)
                drone.target = Target(target_positions[0][0], target_positions[0][1])
                drone.targets_hit = 0

            elif drone.reached_target(drone.target):
                drone.targets_hit += 1
                drone.target = Target(target_positions[drone.targets_hit][0], target_positions[drone.targets_hit][1])

            else:
                drone.time_alive += drone.dt

            # Display Target
            screen.blit(target_image, drone.target.im_pos)
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
