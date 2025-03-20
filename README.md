<!-- uv run python logging_server.py --config_file_path=configs.toml -->



import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import requests
import base64
import os
import sys
from pathlib import Path
import webbrowser
from loguru import logger  # Use loguru for logging
import subprocess

# Add parent directory to path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing functions
from core.body_gestures.detect_pose import detectPose
from core.body_gestures.check_hands_joined import checkHandsJoined
from core.body_gestures.check_horizontal_movement import checkLeftRight
from core.body_gestures.check_vertical_movement import checkJumpCrouch
from utils.config_types import LoggingConfigs
from utils.logging_client import setup_network_logger_client

# Automatically start the logging server when the app runs
def start_logging_server():
    """
    Start the logging server in the background automatically.
    """
    # Path to configs.toml in utils/
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils", "configs.toml"))
    # Path to logging_server.py in utils/
    logging_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils", "logging_server.py"))
    # Ensure the logs directory exists inside utils/
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils", "logs"))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # Start logging_server.py as a background process
    server_process = subprocess.Popen(
        ["python", logging_server_path, "--config_file_path", config_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give the server time to initialize before continuing
    time.sleep(2)  # Slight delay to ensure server is ready
    logger.info("✅ Logging server started successfully.")
    return server_process

# Automatically start the logging server when the app runs
logging_server_process = start_logging_server()

def setup_logging(config_file_path: str) -> None:
    """
    Set up logging using the provided configuration file.
    Args:
        config_file_path: Path to the logging configuration file.
    """
    logging_configs = LoggingConfigs.load_from_path(Path(config_file_path))
    if not logging_configs:
        logger.error("❌ Failed to load logging configurations!")
        return
    # Set up the network logger client with the loaded configurations
    setup_network_logger_client(logging_configs, logger)
    logger.info("✅ Logging initialized successfully.")

class GestureRecognitionApp:
    def __init__(self):
        # Initialize mediapipe pose class for body gestures
        self.mp_pose = mp.solutions.pose
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, 
                                           min_detection_confidence=0.7, min_tracking_confidence=0.7)
        # Initialize mediapipe hands class for hand gestures
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                         max_num_hands=1, 
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7)
        # Hand gesture specific variables
        self.neutral_zone_size = 0.1
        self.neutral_zone_x = 0.5
        self.neutral_zone_y = 0.5
        self.in_neutral_zone = False
        self.current_state = ["neutral"]
        self.debug = True
        # Game state variables
        self.game_started = False
        self.x_pos_index = 1  # Horizontal position index
        self.y_pos_index = 1  # Vertical position index
        self.MID_Y = None     # Initial y-coordinate of shoulders midpoint
        self.counter = 0      # Counter for hands joined frames
        self.num_of_frames = 10  # Number of frames to check for hands joined
        self.time1 = 0        # Previous frame time
        # API client setup
        self.api_endpoints = {
            "FastAPI": "http://localhost:8000",
            "BentoML": "http://localhost:3000"
        }
        self.client_mode = "local"  # "local", "api", or "bentoml"
        self.gesture_type = "body"  # "body" or "hands"
        logger.info("GestureRecognitionApp initialized successfully.")

    def set_api_source(self, api_source):
        """Set the API source to use"""
        if api_source == "FastAPI":
            self.client_mode = "api"
            logger.info("API source set to FastAPI.")
        elif api_source == "BentoML":
            self.client_mode = "bentoml"
            logger.info("API source set to BentoML.")
        else:
            self.client_mode = "local"
            logger.info("Processing mode set to local.")

    def set_gesture_type(self, gesture_type):
        """Set the gesture recognition type"""
        if gesture_type == "Hands":
            self.gesture_type = "hands"
            logger.info("Gesture type set to Hands.")
        else:
            self.gesture_type = "body"
            logger.info("Gesture type set to Body Posture.")

    def reset_state(self):
        """Reset the game state"""
        self.game_started = False
        self.x_pos_index = 1
        self.y_pos_index = 1
        self.MID_Y = None
        self.counter = 0
        self.current_state = ["neutral"]
        logger.info("Game state reset.")
        return "Game state reset"

    def start_game(self):
        """Start the game by clicking at the specified position"""
        if not self.game_started:
            pyautogui.click(x=1300, y=800, button='left')
            self.game_started = True
            logger.info("Game started.")
            return "Game started"
        logger.info("Game already started.")
        return "Game already started"

    def process_frame_local_hands(self, frame):
        """
        Process a frame locally using hand gesture recognition
        Args:
            frame: The input frame from the camera
        Returns:
            Tuple containing processed frame and status info
        """
        logger.info("Processing frame locally using hand gesture recognition.")
        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # Get image dimensions
        image_height, image_width, _ = frame.shape
        status_info = "Status: Processing hand gestures locally\n"
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        # Initialize debug info
        debug_info = None
        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            status_info += "Hand detected: Yes\n"
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                # Detect gestures and control the game
                debug_info = detect_gestures(hand_landmarks)
                # Apply actions based on detected gestures
                if debug_info.action:
                    status_info += f"Action: {debug_info.action}\n"
                    logger.info(f"Detected action: {debug_info.action}")
                    # Perform actions based on gesture detection
                    if debug_info.action == "left":
                        pyautogui.press('left')
                    elif debug_info.action == "right":
                        pyautogui.press('right')
                    elif debug_info.action == "up":
                        pyautogui.press('up')
                    elif debug_info.action == "down":
                        pyautogui.press('down')
                    elif debug_info.action == "space":
                        pyautogui.press('space')
                        # If we're not started, start the game
                        if not self.game_started:
                            self.game_started = True
                            pyautogui.click(x=1300, y=800, button='left')
                # Add detailed info to status
                status_info += f"State: {debug_info.state}\n"
                status_info += f"In neutral: {debug_info.in_neutral}\n"
                status_info += f"Thumb up: {debug_info.thumb_up}\n"
                if debug_info.dx is not None:
                    status_info += f"Movement vector: ({debug_info.dx:.2f}, {debug_info.dy:.2f})\n"
        else:
            status_info += "Hand detected: No\n"
            logger.info("No hand detected.")
        # Draw neutral zone
        center_x = int(image_width * self.neutral_zone_x)
        center_y = int(image_height * self.neutral_zone_y)
        zone_size_x = int(image_width * self.neutral_zone_size)
        zone_size_y = int(image_height * self.neutral_zone_size)
        cv2.rectangle(frame, 
                      (center_x - zone_size_x, center_y - zone_size_y),
                      (center_x + zone_size_x, center_y + zone_size_y),
                      (0, 255, 0) if self.in_neutral_zone else (0, 0, 255),
                      2)
        # Display current state
        cv2.putText(frame, f"State: {self.current_state[0]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Display control instructions
        cv2.putText(frame, "Return to green box between gestures", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Move hand left/right to turn", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Move hand down to slide", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Move hand up to jump", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Thumb up to press space", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Calculate FPS
        time2 = time.time()
        if (time2 - self.time1) > 0:
            fps = 1.0 / (time2 - self.time1)
            status_info += f"FPS: {int(fps)}\n"
            cv2.putText(frame, f'FPS: {int(fps)}', (10, image_height - 30), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        # Update the previous frame time
        self.time1 = time2
        logger.info("Frame processing completed.")
        return frame, status_info

    def process_frame_local_body(self, frame):
        """
        Process a frame locally using body posture recognition
        Args:
            frame: The input frame from the camera
        Returns:
            Tuple containing processed frame and status info
        """
        logger.info("Processing frame locally using body posture recognition.")
        # Flip the frame horizontally for natural (selfie-view) visualization
        frame = cv2.flip(frame, 1)
        # Get the height and width of the frame
        frame_height, frame_width, _ = frame.shape
        # Initialize status info
        status_info = "Status: Processing body posture locally\n"
        # Perform pose detection
        frame, results = detectPose(frame, self.pose_video, draw=self.game_started)
        # Check if pose landmarks are detected
        if results.pose_landmarks:
            status_info += "Pose detected: Yes\n"
            logger.info("Pose detected.")
            # Check if game has started
            if self.game_started:
                status_info += "Game started: Yes\n"
                # Check horizontal movement
                frame, horizontal_position = checkLeftRight(frame, results, draw=True)
                status_info += f"Horizontal position: {horizontal_position}\n"
                logger.info(f"Horizontal position: {horizontal_position}")
                # Check if the person has moved left or right
                if (horizontal_position == 'Left' and self.x_pos_index != 0) or (horizontal_position == 'Center' and self.x_pos_index == 2):
                    # Left movement
                    pyautogui.press('left')
                    self.x_pos_index -= 1
                    status_info += "Action: Move left\n"
                    logger.info("Action: Move left")
                elif (horizontal_position == 'Right' and self.x_pos_index != 2) or (horizontal_position == 'Center' and self.x_pos_index == 0):
                    # Right movement
                    pyautogui.press('right')
                    self.x_pos_index += 1
                    status_info += "Action: Move right\n"
                    logger.info("Action: Move right")
                # Check vertical movement if MID_Y reference is set
                if self.MID_Y:
                    frame, posture = checkJumpCrouch(frame, results, self.MID_Y, draw=True)
                    status_info += f"Vertical position: {posture}\n"
                    logger.info(f"Vertical position: {posture}")
                    # Check if the person has jumped
                    if posture == 'Jumping' and self.y_pos_index == 1:
                        pyautogui.press('up')
                        self.y_pos_index += 1
                        status_info += "Action: Jump\n"
                        logger.info("Action: Jump")
                    # Check if the person has crouched
                    elif posture == 'Crouching' and self.y_pos_index == 1:
                        pyautogui.press('down')
                        self.y_pos_index -= 1
                        status_info += "Action: Crouch\n"
                        logger.info("Action: Crouch")
                    # Check if the person has stood up
                    elif posture == 'Standing' and self.y_pos_index != 1:
                        self.y_pos_index = 1
                        status_info += "Action: Stand\n"
                        logger.info("Action: Stand")
            else:
                status_info += "Game started: No\n"
                logger.info("Game not started.")
                # Show instructions to start the game
                cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            # Check if hands are joined
            frame, hands_status = checkHandsJoined(frame, results)
            status_info += f"Hands status: {hands_status}\n"
            logger.info(f"Hands status: {hands_status}")
            # Handle hands joined action
            if hands_status == 'Hands Joined':
                # Increment counter
                self.counter += 1
                status_info += f"Counter: {self.counter}/{self.num_of_frames}\n"
                logger.info(f"Hands joined counter: {self.counter}/{self.num_of_frames}")
                # Check if counter reaches threshold
                if self.counter == self.num_of_frames:
                    # Start or resume the game
                    if not self.game_started:
                        # Start the game
                        self.game_started = True
                        # Calculate the initial reference for vertical posture
                        left_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
                        right_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
                        self.MID_Y = abs(right_y + left_y) // 2
                        status_info += f"MID_Y reference set: {self.MID_Y}\n"
                        logger.info(f"MID_Y reference set: {self.MID_Y}")
                        # Start the game by clicking at specified position
                        pyautogui.click(x=1300, y=800, button='left')
                        status_info += "Game started\n"
                        logger.info("Game started.")
                    else:
                        # Resume the game after death
                        pyautogui.press('space')
                        status_info += "Game resumed\n"
                        logger.info("Game resumed.")
                    # Reset counter
                    self.counter = 0
            else:
                # Reset counter if hands are not joined
                self.counter = 0
                logger.info("Hands not joined. Counter reset.")
        else:
            status_info += "Pose detected: No\n"
            logger.info("No pose detected.")
            # Reset counter if pose landmarks are not detected
            self.counter = 0
        # Calculate FPS
        time2 = time.time()
        if (time2 - self.time1) > 0:
            fps = 1.0 / (time2 - self.time1)
            status_info += f"FPS: {int(fps)}\n"
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        # Update the previous frame time
        self.time1 = time2
        logger.info("Frame processing completed.")
        return frame, status_info

    def process_frame_local(self, frame):
        """
        Process a frame locally based on the selected gesture type
        Args:
            frame: The input frame from the camera
        Returns:
            Tuple containing processed frame and status info
        """
        if self.gesture_type == "hands":
            return self.process_frame_local_hands(frame)
        else:
            return self.process_frame_local_body(frame)

    def process_frame_api_generic(self, frame, api_type="FastAPI"):
        """
        Process a frame by sending it to the specified API server.
        Args:
            frame: The input frame from the camera.
            api_type: The API type ("FastAPI" or "BentoML").
        Returns:
            Tuple containing processed frame and status info.
        """
        logger.info(f"Starting frame processing via {api_type} API.")
        
        # Get the API endpoint
        api_url = self.api_endpoints.get(api_type, "http://localhost:8000")
        logger.debug(f"Using API URL: {api_url}")
        
        # Flip the frame horizontally for natural (selfie-view) visualization
        frame = cv2.flip(frame, 1)
        logger.debug("Frame flipped horizontally.")
        
        # Encode the frame as base64
        _, buffer = cv2.imencode(".jpg", frame)
        encoded_frame = base64.b64encode(buffer).decode("utf-8")
        logger.debug("Frame encoded as base64.")
        
        # Prepare the payload
        if api_type == "FastAPI":
            # FastAPI expects form data with a file
            files = {
                "file": ("image.jpg", buffer.tobytes(), "image/jpeg"),
            }
            params = {
                "mid_y": self.MID_Y,
                "game_started": self.game_started,
                "gesture_type": self.gesture_type,  # Add gesture type parameter
            }
            logger.debug("Payload prepared for FastAPI request.")
            
            # Send the request to the API
            try:
                logger.info("Sending POST request to FastAPI server.")
                response = requests.post(f"{api_url}/process_frame", files=files, params=params, timeout=5)
                logger.debug(f"FastAPI response received. Status code: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"FastAPI Error: Received status code {response.status_code}")
                    return frame, f"FastAPI Error: Received status code {response.status_code}"
                response_data = response.json()
            except requests.exceptions.ConnectionError:
                logger.error(f"FastAPI Error: Could not connect to server at {api_url}")
                return frame, f"FastAPI Error: Could not connect to server at {api_url}"
            except requests.exceptions.Timeout:
                logger.error("FastAPI Error: Request timed out.")
                return frame, "FastAPI Error: Request timed out"
            except requests.exceptions.RequestException as e:
                logger.error(f"FastAPI Error: {e!s}")
                return frame, f"FastAPI Error: {e!s}"
            except Exception as e:
                logger.error(f"FastAPI Error: {e!s}")
                return frame, f"FastAPI Error: {e!s}"
        
        elif api_type == "BentoML":
            # BentoML expects JSON with base64 encoded image
            payload = {
                "input_data": {
                    "image": encoded_frame,
                    "mid_y": self.MID_Y,
                    "game_started": self.game_started,
                    "gesture_type": self.gesture_type,  # Add gesture type parameter
                },
            }
            logger.debug("Payload prepared for BentoML request.")
            
            # Send the request to the API
            try:
                logger.info("Sending POST request to BentoML server.")
                response = requests.post(f"{api_url}/process_frame", json=payload, timeout=5)
                logger.debug(f"BentoML response received. Status code: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"BentoML Error: Received status code {response.status_code}")
                    return frame, f"BentoML Error: Received status code {response.status_code}"
                
                # Print the raw response for debugging
                logger.debug(f"BentoML API Response: {response.text[:500]}...")
                response_data = response.json()
            except requests.exceptions.ConnectionError:
                logger.error(f"BentoML Error: Could not connect to server at {api_url}")
                return frame, f"BentoML Error: Could not connect to server at {api_url}"
            except requests.exceptions.Timeout:
                logger.error("BentoML Error: Request timed out.")
                return frame, "BentoML Error: Request timed out"
            except requests.exceptions.RequestException as e:
                logger.error(f"BentoML Error: {e!s}")
                return frame, f"BentoML Error: {e!s}"
            except Exception as e:
                logger.error(f"BentoML Error: {e!s}")
                return frame, f"BentoML Error: {e!s}"
        
        else:
            logger.error(f"Unknown API type '{api_type}'")
            return frame, f"Error: Unknown API type '{api_type}'"
        
        # Process the response
        if response_data.get("success", False):
            logger.info("API response indicates success. Processing response data.")
            
            # Decode the processed frame
            try:
                processed_frame = base64.b64decode(response_data["data"]["frame"])
                processed_frame = np.frombuffer(processed_frame, dtype=np.uint8)
                processed_frame = cv2.imdecode(processed_frame, cv2.IMREAD_COLOR)
                
                if processed_frame is None:
                    logger.error(f"{api_type} Error: Failed to decode processed frame.")
                    return frame, f"{api_type} Error: Failed to decode processed frame."
                
                # Extract results
                results = response_data["data"].get("results", {})
                logger.debug(f"Extracted results: {results}")
                
                # Handle different detection types
                if self.gesture_type == "hands":
                    # Update hand gesture state
                    if "current_state" in response_data["data"]:
                        self.current_state[0] = response_data["data"]["current_state"]
                        logger.debug(f"Updated current state: {self.current_state[0]}")
                    
                    # Handle hand gesture actions
                    if results.get("action"):
                        action = results["action"]
                        logger.info(f"Detected hand gesture action: {action}")
                        
                        if action == "left":
                            pyautogui.press("left")
                        elif action == "right":
                            pyautogui.press("right")
                        elif action == "up":
                            pyautogui.press("up")
                        elif action == "down":
                            pyautogui.press("down")
                        elif action == "space":
                            pyautogui.press("space")
                            # Start game if not started
                            if not self.game_started:
                                self.game_started = True
                                pyautogui.click(x=1300, y=800, button="left")
                                logger.info("Game started via space gesture.")
                else:
                    # Body posture detection
                    pose_detected = response_data["data"].get("pose_detected", False)
                    logger.debug(f"Pose detected: {pose_detected}")
                    
                    if pose_detected:
                        # Check if hands are joined
                        hands_joined = results.get("hands_joined", False)
                        logger.debug(f"Hands joined: {hands_joined}")
                        
                        if hands_joined:
                            # Increment counter
                            self.counter += 1
                            logger.debug(f"Hands joined counter incremented: {self.counter}/{self.num_of_frames}")
                            
                            # Check if counter reaches threshold
                            if self.counter == self.num_of_frames:
                                # Start or resume the game
                                if not self.game_started:
                                    # Start the game
                                    self.game_started = True
                                    # Calculate the initial reference for vertical posture
                                    # Try to get the mid_y from the API if available
                                    try:
                                        if api_type == "FastAPI":
                                            calc_files = {
                                                "file": ("image.jpg", buffer.tobytes(), "image/jpeg"),
                                            }
                                            calc_response = requests.post(f"{api_url}/calculate_reference",
                                                                        files=calc_files, timeout=5)
                                        else:  # BentoML
                                            calc_payload = {"input_data": {"image": encoded_frame}}
                                            calc_response = requests.post(f"{api_url}/calculate_reference",
                                                                        json=calc_payload, timeout=5)
                                        calc_data = calc_response.json()
                                        if calc_data.get("success", False):
                                            self.MID_Y = calc_data["data"].get("mid_y")
                                            logger.info(f"MID_Y calculated via API: {self.MID_Y}")
                                    except Exception as e:
                                        # Fall back to local calculation
                                        logger.error(f"Error calculating reference: {e!s}")
                                        frame_height = frame.shape[0]
                                        # Assume an average value for demo purposes
                                        self.MID_Y = frame_height // 2
                                        logger.warning(f"Fallback MID_Y set locally: {self.MID_Y}")
                                    
                                    # Start the game by clicking at specified position
                                    pyautogui.click(x=1300, y=800, button="left")
                                    logger.info("Game started via hands joined gesture.")
                                else:
                                    # Resume the game after death
                                    pyautogui.press("space")
                                    logger.info("Game resumed via hands joined gesture.")
                                
                                # Reset counter
                                self.counter = 0
                                logger.debug("Counter reset after game start/resume.")
                        else:
                            # Reset counter if hands are not joined
                            self.counter = 0
                            logger.debug("Counter reset as hands are no longer joined.")
                        
                        # Process game controls if game is started
                        if self.game_started and self.MID_Y:
                            # Check horizontal movement
                            horizontal_position = results.get("horizontal_position")
                            if horizontal_position:
                                logger.debug(f"Horizontal position detected: {horizontal_position}")
                                
                                # Check if the person has moved left or right
                                if (horizontal_position == "Left" and self.x_pos_index != 0) or (horizontal_position == "Center" and self.x_pos_index == 2):
                                    # Left movement
                                    pyautogui.press("left")
                                    self.x_pos_index -= 1
                                    logger.info("Action: Move left.")
                                elif (horizontal_position == "Right" and self.x_pos_index != 2) or (horizontal_position == "Center" and self.x_pos_index == 0):
                                    # Right movement
                                    pyautogui.press("right")
                                    self.x_pos_index += 1
                                    logger.info("Action: Move right.")
                            
                            # Check vertical movement
                            vertical_position = results.get("vertical_position")
                            if vertical_position:
                                logger.debug(f"Vertical position detected: {vertical_position}")
                                
                                # Check if the person has jumped
                                if vertical_position == "Jumping" and self.y_pos_index == 1:
                                    pyautogui.press("up")
                                    self.y_pos_index += 1
                                    logger.info("Action: Jump.")
                                # Check if the person has crouched
                                elif vertical_position == "Crouching" and self.y_pos_index == 1:
                                    pyautogui.press("down")
                                    self.y_pos_index -= 1
                                    logger.info("Action: Crouch.")
                                # Check if the person has stood up
                                elif vertical_position == "Standing" and self.y_pos_index != 1:
                                    self.y_pos_index = 1
                                    logger.info("Action: Stand.")
                
                # Prepare status info
                status_info = f"Status: Processing frame via {api_type} API\n"
                status_info += f"API URL: {api_url}\n"
                status_info += f"Game started: {self.game_started}\n"
                status_info += f"Gesture type: {self.gesture_type}\n"
                
                # Add gesture type specific info
                if self.gesture_type == "hands":
                    status_info += f"Current state: {self.current_state[0]}\n"
                else:
                    status_info += f"Pose detected: {response_data['data'].get('pose_detected', False)}\n"
                    status_info += f"MID_Y: {self.MID_Y}\n"
                    status_info += f"Counter: {self.counter}/{self.num_of_frames}\n"
                
                # Add results info
                if results:
                    status_info += "Results:\n"
                    for key, value in results.items():
                        status_info += f"  {key}: {value}\n"
                
                logger.info("Frame processing completed successfully.")
                return processed_frame, status_info
            
            except Exception as e:
                logger.error(f"Error during frame processing: {e!s}")
                return frame, f"{api_type} Error in processing response: {e!s}"
        else:
            # Log detailed error from the API if available
            error_msg = response_data.get("message", "Unknown error")
            logger.error(f"{api_type} API returned an error: {error_msg}")
            return frame, f"{api_type} API Error: {error_msg}"
        
def process_frame(self, frame):
    """
    Process a frame based on the selected client mode.
    
    Args:
        frame: The input frame from the camera.
        
    Returns:
        Tuple containing processed frame and status info.
    """
    logger.info("Processing frame based on client mode.")
    
    if self.client_mode == "local":
        logger.debug("Client mode set to 'local'. Processing frame locally.")
        return self.process_frame_local(frame)
    elif self.client_mode == "api":
        logger.debug("Client mode set to 'api'. Processing frame via FastAPI.")
        return self.process_frame_api_generic(frame, api_type="FastAPI")
    elif self.client_mode == "bentoml":
        logger.debug("Client mode set to 'bentoml'. Processing frame via BentoML.")
        return self.process_frame_api_generic(frame, api_type="BentoML")
    else:
        logger.error(f"Invalid client mode: {self.client_mode}")
        return frame, "Error: Invalid client mode"

# Initialize the app
app = GestureRecognitionApp()

# Streamlit App
def main():
    st.title("Gesture Control Game 🎮")
    logger.info("Starting Streamlit app.")

    # Step 1: Select API Source
    st.header("Step 1: Select API Source")
    api_source = st.selectbox("Choose the API source:", ["Local Processing", "FastAPI", "BentoML"])
    logger.info(f"User selected API source: {api_source}")

    # Update the app's API source
    if api_source != "Local Processing":
        logger.debug(f"Setting API source to: {api_source}")
        app.set_api_source(api_source)
    else:
        logger.debug("Setting client mode to 'local'.")
        app.client_mode = "local"

    st.info(f"Using {api_source} for gesture recognition")

    # Step 2: Select Gesture Recognition Type
    st.header("Step 2: Select Gesture Recognition Type")
    gesture_type = st.radio("Choose the gesture recognition type:", ["Body Posture", "Hands"])
    logger.info(f"User selected gesture recognition type: {gesture_type}")
    
    app.set_gesture_type("Hands" if gesture_type == "Hands" else "Body Posture")
    logger.debug(f"Gesture type set to: {app.gesture_type}")

    # Display appropriate instructions based on gesture type
    if gesture_type == "Body Posture":
        st.subheader("Body Posture Controls")
        st.markdown("""
        - **Start/Resume Game**: Join both hands together for a few seconds
        - **Move Left/Right**: Lean your body left or right
        - **Jump**: Stand on your tiptoes or jump slightly
        - **Crouch/Slide**: Bend your knees or crouch down
        """)
    else:  # Hands
        st.subheader("Hand Gesture Controls")
        st.markdown("""
        - **Start/Resume Game**: Thumb up gesture
        - **Move Left/Right**: Move hand left or right from neutral zone
        - **Jump**: Move hand up from neutral zone
        - **Crouch/Slide**: Move hand down from neutral zone
        - **Return to neutral zone (green box) between gestures**
        """)

    # Step 3: Choose Game
    st.header("Step 3: Choose Game")
    game_options = {
        "Subway Surfers": "https://poki.com/en/g/subway-surfers",
        "Moto X3M": "https://poki.com/en/g/moto-x3m",
        "Temple Run 2": "https://poki.com/en/g/temple-run-2",
    }
    selected_game = st.selectbox("Select a game to play:", list(game_options.keys()))
    logger.info(f"User selected game: {selected_game}")

    # Open the selected game in a new browser tab
    if st.button("Play Selected Game"):
        logger.info(f"Opening game URL: {game_options[selected_game]}")
        webbrowser.open_new_tab(game_options[selected_game])

    # Step 4: Start Webcam
    st.header("Step 4: Start Webcam")
    run_webcam = st.checkbox("Start Webcam")
    logger.debug(f"Webcam checkbox state: {run_webcam}")

    # Configure API settings (could be loaded from config file)
    if api_source != "Local Processing":
        with st.expander("API Settings"):
            fastapi_url = st.text_input("FastAPI URL", value="http://localhost:8000")
            bentoml_url = st.text_input("BentoML URL", value="http://localhost:3000")

            # Update API endpoints
            app.api_endpoints["FastAPI"] = fastapi_url
            app.api_endpoints["BentoML"] = bentoml_url
            logger.debug(f"Updated API endpoints: {app.api_endpoints}")

    # Display webcam feed
    if run_webcam:
        # Allow user to reset game state
        if st.button("Reset Game State"):
            logger.info("Resetting game state.")
            message = app.reset_state()
            st.success(message)

        # Create placeholders for frame and status
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera.")
            st.error("Failed to open camera.")
        else:
            try:
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to capture frame from webcam.")
                        st.error("Failed to capture frame from webcam.")
                        break

                    # Process the frame
                    logger.debug("Processing frame from webcam.")
                    processed_frame, status_info = app.process_frame(frame)

                    # Display the processed frame
                    frame_placeholder.image(processed_frame, channels="BGR")

                    # Display status info
                    status_placeholder.text(status_info)

            except Exception as e:
                logger.exception(f"Error processing video: {e}")
                st.error(f"Error processing video: {e}")

            finally:
                # Release the webcam
                if cap and cap.isOpened():
                    logger.info("Releasing webcam and destroying OpenCV windows.")
                    cap.release()
                    cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == "__main__":
    logger.info("Starting the application.")
    main()