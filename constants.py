# Game constants
FPS = 60
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
SPEED_UP = 1

GRAVITY = .08 * SPEED_UP

# Propeller force for UP and DOWN
THRUST_AMP = 0.04 * SPEED_UP
# Propeller force for LEFT and RIGHT rotations
DIFF_AMP = 0.003 * SPEED_UP

DRONE_PATH = "assets/images/drone_img.png"
DRONE_WIDTH = 200 #px
DRONE_LENGTH = 30 #px

TARGET_PATH = "assets/images/red_target.png"
TARGET_WIDTH = 100
NUM_TARGETS = 5

HIT_THRESH = 50

INTERACTIVE = False

UPDATES_PER_STEP = 10

MAX_DIST = 1200