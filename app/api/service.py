

import base64
import os
import sys
from typing import Any

import bentoml
import cv2
import mediapipe as mp
import numpy as np
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loguru import logger

with bentoml.importing():
    from core.body_gestures.check_hands_joined import checkHandsJoined
    from core.body_gestures.check_horizontal_movement import checkLeftRight
    from core.body_gestures.check_vertical_movement import checkJumpCrouch
    from core.body_gestures.detect_pose import detectPose
    from core.hand_gestures.detect_gestures import detect_gestures

    # Import hand gesture functions

# Define request model
class ProcessFrameRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    mid_y: int | None = Field(None, description="Reference Y-coordinate for vertical movement detection")
    game_started: bool = Field(False, description="Whether the game has started")
    gesture_type: str = Field("body", description="Type of gesture recognition (body or hands)")

# Define response model
class GestureResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] | None = None

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Initialize mediapipe hands class
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Define a BentoML service using the v0.x syntax
@bentoml.service
class GestureRecognitionService:
    def __init__(self):
        logger.info("Initializing GestureRecognitionService...")

        # Initialize MediaPipe models
        self.pose_video = pose_video
        self.hands = hands
        self.mp_drawing = mp_drawing
        self.mp_hands = mp_hands

        # Hand gesture specific variables
        self.neutral_zone_size = 0.1
        self.neutral_zone_x = 0.5
        self.neutral_zone_y = 0.5
        self.in_neutral_zone = False
        self.current_state = "neutral"

    @bentoml.api
    def process_frame(self, input_data: ProcessFrameRequest) -> dict:
        """Process a video frame and return gesture recognition results.
        """
        try:
            logger.info(f"Processing frame. Gesture Type: {input_data.gesture_type}, Game Started: {input_data.game_started}")
            # Read the image from base64
            encoded_image = input_data.image
            image_bytes = base64.b64decode(encoded_image)
            frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logger.error("Invalid image data received")
                return {
                    "success": False,
                    "message": "Invalid image data or image could not be decoded",
                }
            # Get the gesture type
            gesture_type = input_data.gesture_type  # Access as an attribute
            # Process the frame based on gesture type
            if gesture_type == "hands":
                return self.process_hand_gestures(frame, input_data)
            return self.process_body_posture(frame, input_data)
        except Exception as e:
            logger.error(f"Error processing frame: {e!s}")
            return {
                "success": False,
                "message": f"Error processing frame: {e!s}",
            }

    def process_hand_gestures(self, frame, input_data: ProcessFrameRequest):
        """Process frame using hand gesture recognition"""
        try:
            logger.info("Processing hand gestures...")
            # Flip the image horizontally for a selfie-view display
            # frame = cv2.flip(frame, 1)
            # Get image dimensions
            image_height, image_width, _ = frame.shape
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image and detect hands
            results = self.hands.process(image_rgb)
            # Initialize action and debug info
            action = None
            debug_info = None
            # Draw hand landmarks on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )
                    # Detect gestures and control the game
                    debug_info = detect_gestures(hand_landmarks)
                    # Store the action
                    if debug_info.action:
                        action = debug_info.action
            # Draw neutral zone
            center_x = int(image_width * self.neutral_zone_x)
            center_y = int(image_height * self.neutral_zone_y)
            zone_size_x = int(image_width * self.neutral_zone_size)
            zone_size_y = int(image_height * self.neutral_zone_size)
            in_neutral = debug_info.in_neutral if debug_info else False
            cv2.rectangle(frame,
                        (center_x - zone_size_x, center_y - zone_size_y),
                        (center_x + zone_size_x, center_y + zone_size_y),
                        (0, 255, 0) if in_neutral else (0, 0, 255),
                        2)
            # Update current state
            if debug_info:
                self.current_state = debug_info.state
            # Display current state
            cv2.putText(frame, f"State: {self.current_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Encode the processed frame as base64
            _, buffer = cv2.imencode(".jpg", frame)
            encoded_frame = base64.b64encode(buffer).decode("utf-8")

            logger.info(f"Hand gesture processing completed. Action: {action}")
            # Prepare the results dictionary
            frame_results = {}
            if debug_info:
                frame_results = {
                    "action": action,
                    "state": debug_info.state,
                    "in_neutral": debug_info.in_neutral,
                    "thumb_up": debug_info.thumb_up,
                }
                # Add movement vector if available
                if debug_info.dx is not None:
                    frame_results["dx"] = debug_info.dx
                    frame_results["dy"] = debug_info.dy
            return {
                "success": True,
                "message": "Frame processed successfully with hand gestures",
                "data": {
                    "frame": encoded_frame,
                    "results": frame_results,
                    "current_state": self.current_state,
                },
            }
        except Exception as e:
            logger.error(f"Error processing hand gestures: {e!s}")
            return {
                "success": False,
                "message": f"Error processing hand gestures: {e!s}",
            }

    def process_body_posture(self, frame, input_data: ProcessFrameRequest):
        """Process frame using body posture recognition"""
        try:
            logger.info("Processing body posture...")
            # Flip the frame horizontally for a selfie-view display
            # frame = cv2.flip(frame, 1)
            # Get game state from input
            game_started = input_data.game_started  # Access as an attribute
            # Detect pose
            processed_frame, results = detectPose(frame, self.pose_video, draw=game_started)
            # Prepare results dictionary
            frame_results = {}
            # Check if pose landmarks are detected
            if results.pose_landmarks:
                # Check hands joined
                _, hands_status = checkHandsJoined(processed_frame, results)
                frame_results["hands_joined"] = (hands_status == "Hands Joined")
                # Check horizontal movement if game is started
                if game_started:
                    _, horizontal_position = checkLeftRight(processed_frame, results, draw=True)
                    frame_results["horizontal_position"] = horizontal_position
                    # Check vertical movement if mid_y is provided
                    mid_y = input_data.mid_y  # Access as an attribute
                    if mid_y:
                        _, posture = checkJumpCrouch(processed_frame, results, mid_y, draw=True)
                        frame_results["vertical_position"] = posture
            # Encode the processed frame as base64
            _, buffer = cv2.imencode(".jpg", processed_frame)
            encoded_frame = base64.b64encode(buffer).decode("utf-8")
            logger.info("Body posture processing completed successfully.")

            return {
                "success": True,
                "message": "Frame processed successfully with body posture",
                "data": {
                    "frame": encoded_frame,
                    "pose_detected": results.pose_landmarks is not None,
                    "results": frame_results,
                },
            }
        except Exception as e:
            logger.error(f"Error processing body posture: {e!s}")
            return {
                "success": False,
                "message": f"Error processing body posture: {e!s}",
            }

    @bentoml.api
    def calculate_reference(self, input_data: dict) -> dict:
        """Calculate the reference Y-coordinate for vertical movement detection.
        """
        try:
            logger.info("Calculating reference Y-coordinate...")
            # Read the image from the provided base64 data
            encoded_image = input_data["image"]
            image_bytes = base64.b64decode(encoded_image)
            frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Invalid image data for reference calculation.")
                return {
                    "success": False,
                    "message": "Invalid image data or image could not be decoded",
                }

            # Detect pose
            _, results = detectPose(frame, self.pose_video, draw=False)

            # Check if pose landmarks are detected
            if not results.pose_landmarks:
                logger.warning("No pose landmarks detected for reference calculation.")
                return {
                    "success": False,
                    "message": "No pose landmarks detected in the frame",
                }

            # Calculate the mid-Y reference
            frame_height = frame.shape[0]
            left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
            right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
            mid_y = abs(right_y + left_y) // 2
            logger.info(f"Reference Y-coordinate calculated: {mid_y}")

            return {
                "success": True,
                "message": "Reference Y-coordinate calculated successfully",
                "data": {"mid_y": mid_y},
            }
        except Exception as e:
            logger.error(f"Error calculating reference: {e!s}")
            return {
                "success": False,
                "message": f"Error calculating reference: {e!s}",
            }
