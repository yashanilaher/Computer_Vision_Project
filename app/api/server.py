from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from typing import Dict, Any, Optional
import sys
import os


# Add parent directory to path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from utils.config_types import LoggingConfigs
from utils.logging_client import setup_network_logger_client

# Import your existing body gesture functions
from core.body_gestures.detect_pose import detectPose
from core.body_gestures.check_hands_joined import checkHandsJoined
from core.body_gestures.check_horizontal_movement import checkLeftRight
from core.body_gestures.check_vertical_movement import checkJumpCrouch

# Import hand gesture functions
from core.hand_gestures.initialize import initialize_hands
from core.hand_gestures.detect_gestures import detect_gestures
import mediapipe as mp



# Initialize FastAPI app
app = FastAPI(
    title="Gesture Recognition API",
    description="API for gesture recognition for gaming applications",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize mediapipe pose class for body gestures
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe hands class for hand gestures
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Define response model
class GestureResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Hand gesture processing parameters
neutral_zone_size = 0.1
neutral_zone_x = 0.5
neutral_zone_y = 0.5

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Gesture Recognition API is running"}

@app.post("/process_frame", response_model=GestureResponse)
async def process_frame(file: UploadFile = File(...), 
                        mid_y: Optional[int] = None, 
                        game_started: bool = False,
                        gesture_type: str = "body"):
    """
    Process a video frame and return gesture recognition results.
    
    - **file**: The image file (frame) to process
    - **mid_y**: Reference Y-coordinate for vertical movement detection
    - **game_started**: Whether the game has started
    - **gesture_type**: Type of gesture to detect ("body" or "hands")
    """
    try:
        logger.info(f"Processing frame. Game started: {game_started}, MID_Y: {mid_y}, Gesture type: {gesture_type}")
        
        # Read the image file
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            ("Invalid image file received")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Flip the frame horizontally for selfie-view
        # frame = cv2.flip(frame, 1)
        
        # Check which gesture type to process
        if gesture_type.lower() == "hands":
            return await process_hand_gestures(frame, game_started)
        else:
            return await process_body_gestures(frame, mid_y, game_started)
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return GestureResponse(
            success=False,
            message=f"Error processing frame: {str(e)}"
        )

async def process_body_gestures(frame, mid_y, game_started):
    """Process body gestures in the frame"""
    try:
        logger.info("Processing body gestures...")
        # Process the frame
        frame_results = {}
        
        # Detect pose
        processed_frame, results = detectPose(frame, pose_video, draw=game_started)
        
        # Check if pose landmarks are detected
        if results.pose_landmarks:
            logger.debug("Pose landmarks detected")
            
            # Check hands joined
            processed_frame, hands_status = checkHandsJoined(processed_frame, results)
            frame_results["hands_joined"] = (hands_status == "Hands Joined")
            
            # Check horizontal movement
            if game_started:
                processed_frame, horizontal_position = checkLeftRight(processed_frame, results, draw=True)
                frame_results["horizontal_position"] = horizontal_position
                
                # Check vertical movement if mid_y is provided
                if mid_y:
                    processed_frame, posture = checkJumpCrouch(processed_frame, results, mid_y, draw=True)
                    frame_results["vertical_position"] = posture
        else:
            logger.warning("No pose landmarks detected.")
        
        # Encode the processed frame as base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("Body gesture frame processed successfully.")

        # Return the results
        return GestureResponse(
            success=True,
            message="Body gesture frame processed successfully",
            data={
                "frame": encoded_frame,
                "pose_detected": results.pose_landmarks is not None,
                "results": frame_results
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing body gestures: {str(e)}")
        return GestureResponse(
            success=False,
            message=f"Error processing body gestures: {str(e)}"
        )

async def process_hand_gestures(frame, game_started):
    """Process hand gestures in the frame"""
    try:
        logger.info("Processing hand gestures...")
        # Get image dimensions
        image_height, image_width, _ = frame.shape
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        frame_results = {}
        current_state = "neutral"
        action = None
        thumb_up = False
        dx = None
        dy = None
        in_neutral = False
        
        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Detect gestures and get detection results
                debug_info = detect_gestures(hand_landmarks)
                
                # Update results
                frame_results["action"] = debug_info.action
                action = debug_info.action
                current_state = debug_info.state
                thumb_up = debug_info.thumb_up
                in_neutral = debug_info.in_neutral
                dx = debug_info.dx
                dy = debug_info.dy
        
        # Draw neutral zone
        center_x = int(image_width * neutral_zone_x)
        center_y = int(image_height * neutral_zone_y)
        zone_size_x = int(image_width * neutral_zone_size)
        zone_size_y = int(image_height * neutral_zone_size)
        cv2.rectangle(frame, 
                      (center_x - zone_size_x, center_y - zone_size_y),
                      (center_x + zone_size_x, center_y + zone_size_y),
                      (0, 255, 0) if in_neutral else (0, 0, 255),
                      2)
        
        # Display current state
        cv2.putText(frame, f"State: {current_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode the processed frame as base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        logger.info("Hand gesture frame processed successfully.")
        # Return the results
        return GestureResponse(
            success=True,
            message="Hand gesture frame processed successfully",
            data={
                "frame": encoded_frame,
                "hand_detected": results.multi_hand_landmarks is not None,
                "current_state": current_state,
                "results": {
                    "action": action,
                    "thumb_up": thumb_up,
                    "in_neutral": in_neutral,
                    "dx": dx,
                    "dy": dy
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing hand gestures: {str(e)}")
        return GestureResponse(
            success=False,
            message=f"Error processing hand gestures: {str(e)}"
        )

@app.post("/calculate_reference", response_model=GestureResponse)
async def calculate_reference(file: UploadFile = File(...)):
    """
    Calculate the reference Y-coordinate for vertical movement detection.
    
    - **file**: The image file (frame) to process
    """
    try:
        logger.info("Calculating reference Y-coordinate")
        
        # Read the image file
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Invalid image file received")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Flip the frame horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        
        # Detect pose
        _, results = detectPose(frame, pose_video, draw=False)
        
        # Check if pose landmarks are detected
        if not results.pose_landmarks:
            logger.warning("No pose landmarks detected for reference calculation")
            return GestureResponse(
                success=False,
                message="No pose landmarks detected in the frame"
            )
        
        # Calculate the mid-Y reference
        frame_height = frame.shape[0]
        left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
        right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
        mid_y = abs(right_y + left_y) // 2
        
        logger.info(f"Reference Y-coordinate calculated: {mid_y}")
        
        return GestureResponse(
            success=True,
            message="Reference Y-coordinate calculated successfully",
            data={"mid_y": mid_y}
        )
    
    except Exception as e:
        logger.error(f"Error calculating reference: {str(e)}")
        return GestureResponse(
            success=False,
            message=f"Error calculating reference: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)