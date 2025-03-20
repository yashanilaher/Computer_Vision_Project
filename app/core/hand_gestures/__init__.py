import mediapipe as mp
import time
import cv2
from .pydantic_models import HandPositionsModel

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Game control settings
jump_threshold = 0.15  # Threshold for jump gesture (upward movement)
left_threshold = 0.15  # Threshold for left movement
right_threshold = 0.15  # Threshold for right movement
down_threshold = 0.15  # Threshold for down/slide movement

# Neutral zone settings
neutral_zone_size = 0.1  # Size of neutral zone
in_neutral_zone = [True]   # Start in neutral zone
neutral_zone_x = 0.5     # Center X of neutral zone
neutral_zone_y = 0.5     # Center Y of neutral zone

# Hand position reference when entering neutral zone
reference_hand_x = [None]
reference_hand_y = [None]

# State tracking
current_state = ["neutral"]  # Current gesture state
state_start_time = [time.time()]  # When the current state began

# Movement tracking
hand_positions_data =  HandPositionsModel(hand_positions=[]) # Store recent hand positions
position_history_length = 5  # Number of frames to keep

# Cooldown for actions to prevent rapid firing
last_action_time = [time.time()]
cooldown = [0.3]  # seconds

# Thumb up gesture settings
thumb_up_cooldown = [1.0]  # longer cooldown for thumb up gesture
last_thumb_up_time = [0]  # Last time thumb up was detected

# Debug flag
debug = True

# # Camera setup
# cap = cv2.VideoCapture(0)