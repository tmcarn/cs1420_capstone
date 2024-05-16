import pygame
import numpy as np
import argparse
import matplotlib.pyplot as plt


from Drones import HumanDrone, PIDDrone, SACDrone, PIDSACDrone, PIDSAC2Drone
from target import Target
import constants
   
def main(args):
    if args.interactive:
        constants.INTERACTIVE = True
    if args.vis_thrust:
        vis_thrust = True
    else: 
        vis_thrust = False

    

    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    total_time = 0

    # Load drone image
    drone_image = pygame.image.load(constants.DRONE_PATH)
    # Load target image
    target_image = pygame.image.load(constants.TARGET_PATH)
    font = pygame.font.SysFont('Arial', 12)

    # Generate targets
    if constants.INTERACTIVE:
        target_positions = np.random.randint(constants.SCREEN_HEIGHT, size=(constants.NUM_TARGETS, 2))
    else:
        target_positions = np.random.randint(constants.SCREEN_HEIGHT, size=(constants.NUM_TARGETS, 2))

    drones = [PIDDrone(screen.get_width()/2, screen.get_height()/2),
              SACDrone(screen.get_width()/2, screen.get_height()/2, load_from='oscar_models/SAC 5_300_000.zip'),
              PIDSACDrone(screen.get_width()/2, screen.get_height()/2, load_from='oscar_models/pid 280_000.zip'),
              PIDSAC2Drone(screen.get_width()/2, screen.get_height()/2, load_from='oscar_models/pid3.0 100_000.zip')]

    for drone in drones:
        # Update Target
        drone.target = Target(target_positions[0][0], target_positions[0][1])

    # Main loop
    running = True
    while running:
        screen.fill((200, 200, 200))

        # Quitable
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if constants.INTERACTIVE:
            # Update target position with mouse movement
            target_x, target_y = pygame.mouse.get_pos()
            global_target = Target(target_x, target_y) # Only interact with one target

        for i,drone in enumerate(drones):

            if constants.INTERACTIVE:
                drone.target = global_target

            new_thrust = drone.act(drone.target) 
            drone.update(new_thrust)

            if drone.is_dead():
                drone.respawn(screen.get_width()/2, screen.get_height()/2)
                drone.target = Target(target_positions[0][0], target_positions[0][1])
                drone.targets_hit = 0

            elif drone.reached_target(drone.target) and not drone.finished:
                if not constants.INTERACTIVE:
                    drone.targets_hit += 1
                    if drone.targets_hit == constants.NUM_TARGETS:
                        drone.finished = True
                        drone.time_to_finish = total_time
                        break # All Done
                    drone.target = Target(target_positions[drone.targets_hit][0], target_positions[drone.targets_hit][1])
                    
                    if drone.name=='PIDSAC2':
                        drone.update_pid(drone.target)
            else:
                drone.time_alive += drone.dt

            # Display Target
            screen.blit(target_image, drone.target.im_pos)
            # Display Drone
            drone_name = font.render(drone.name, True, (0, 0, 0))
            
            drone_score = font.render(f"{drone.name} score: {drone.targets_hit}", True, (0,0,0))
            curr_drone = drone_image.copy()
            curr_drone.blit(drone_name, (50,50))
            rotated_image = pygame.transform.rotate(curr_drone, drone.pos[2])
            rotated_rect = rotated_image.get_rect(center=curr_drone.get_rect(topleft=drone.im_pos).center)

            screen.blit(rotated_image, rotated_rect)
            screen.blit(drone_score, ((i+1)*150, 0))
            if drone.finished:
                finish_text = font.render(f"Final Time: {drone.time_to_finish:.2f}s", True, (24, 100, 16))
                screen.blit(finish_text, ((i+1)*150, 12))

            # Visualize Thrust in Running Plot
            if vis_thrust:
                pygame.draw.rect(screen, (24, 100, 16), ((i+1)*150, 24, 25, new_thrust[0]*1000))
                pygame.draw.rect(screen, (24, 100, 16), (((i+1)*150)+35, 24, 25, new_thrust[1]*1000))

            drone.dt = 1/constants.FPS

        # Cap the frame rate
        clock_tick = clock.tick(constants.FPS) / 1000
        total_time += clock_tick

        total_time_text = font.render(f"Total Time Elapsed: {total_time:.2f}s", True, (0,0,0))
        screen.blit(total_time_text, (0, 0))

        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()


if __name__=="__main__":
    # call the main function
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('-v', '--vis_thrust', action='store_true', help='Visualize Thrust')
    args = parser.parse_args()
    main(args)




