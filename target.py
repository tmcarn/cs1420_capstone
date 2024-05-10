import numpy as np
import constants


class Target():
    def __init__(self, x, y) -> None:
        self.pos = np.array([x, y])
        self.im_pos = (x - constants.TARGET_WIDTH//2, y - constants.TARGET_WIDTH//2)

        self.hit = False
    
    def update_pos(self, x,y):
        self.pos = np.array([x, y])
        self.im_pos = (x - constants.TARGET_WIDTH//2, y - constants.TARGET_WIDTH//2)
