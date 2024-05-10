import pygame
import numpy as np

from player import HumanDrone, PIDDrone, PPODrone, SACDrone
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

    drones = [SACDrone(screen.get_width()/2, screen.get_height()/2)]

    for drone in drones:
        drone.assign_new_target(target_positions[0][0], target_positions[0][1])

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
                drone.current_target.update_pos(target_x, target_y) # Only interact with one target

        for drone in drones:

            i = drone.target_num

            # drone.assign_new_target(target_positions[i][0], target_positions[i][1])

            new_thrust = drone.act() 
            drone.update(new_thrust)

            if drone.is_dead():
                drone.respawn()
                drone.assign_new_target(target_positions[0][0], target_positions[0][1])

            elif drone.reached_target():
                if not constants.INTERACTIVE: 
                    drone.target_num += 1
                    drone.assign_new_target(target_positions[i][0], target_positions[i][1])

            else:
                drone.time_alive += drone.dt

            # Display Target
            screen.blit(target_image, drone.current_target.im_pos)
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
