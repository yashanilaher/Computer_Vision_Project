import cv2
import mediapipe as mp
from loguru import logger
from utils.config_types import LoggingConfigs
from utils.logging_client import setup_network_logger_client

from . import cap, hands, mp_hands, mp_drawing
from .detect_gestures import detect_gestures
from . import neutral_zone_size, neutral_zone_x, neutral_zone_y, in_neutral_zone, current_state, debug
from .pydantic_models import GestureDetectionResult


def run():
    """
    Run the hand gesture game controller.
    """
    logger.info("Starting hand gesture game controller...")

    try:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                logger.warning("Ignoring empty camera frame. No frame captured.")
                continue

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and detect hand landmarks
            results = hands.process(image_rgb)
            logger.debug("Hand landmarks processed successfully.")

            # Get image dimensions
            image_height, image_width, _ = image.shape

            # Initialize debug info
            debug_info = GestureDetectionResult(
                hand_x=0, hand_y=0, in_neutral=False, state="neutral", thumb_up=False, action=None, dx=None, dy=None
            )

            # Draw hand landmarks on the image
            if results.multi_hand_landmarks:
                logger.info(f"Detected {len(results.multi_hand_landmarks)} hand(s)")

                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    logger.debug("Hand landmarks drawn on the image.")

                    # Detect gestures and control the game
                    debug_info = detect_gestures(hand_landmarks)
                    logger.info(f"Gesture detected: {debug_info.action}")

            # Draw neutral zone
            center_x = int(image_width * neutral_zone_x)
            center_y = int(image_height * neutral_zone_y)
            zone_size_x = int(image_width * neutral_zone_size)
            zone_size_y = int(image_height * neutral_zone_size)

            cv2.rectangle(
                image,
                (center_x - zone_size_x, center_y - zone_size_y),
                (center_x + zone_size_x, center_y + zone_size_y),
                (0, 255, 0) if in_neutral_zone else (0, 0, 255),
                2
            )
            logger.debug("Neutral zone drawn on the image.")

            # Display current state
            cv2.putText(image, f"State: {current_state[0]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display control instructions
            instructions = [
                "Return to green box between gestures",
                "Move hand left/right to turn",
                "Move hand down to slide",
                "Move hand up to jump",
                "Thumb up to press space"
            ]

            y_pos = 60
            for instruction in instructions:
                cv2.putText(image, instruction, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30

            logger.debug("Control instructions displayed.")

            # Display debug information if enabled
            if debug and debug_info:
                logger.debug("Displaying debug information...")
                y_pos = 210
                for key, value in debug_info.dict().items():
                    if isinstance(value, float):
                        value = round(value, 3)
                    text = f"{key}: {value}"
                    cv2.putText(image, text, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 25

            # Show the image
            cv2.imshow('Hand Gesture Game Controller', image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                logger.info("Exiting application. 'q' key pressed.")
                break

    except Exception as e:
        logger.error(f"Error occurred while running the game: {e}")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera and windows released. Application terminated.")
