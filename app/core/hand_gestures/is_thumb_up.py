from loguru import logger
from utils.config_types import LoggingConfigs
from utils.logging_client import setup_network_logger_client
from . import mp_hands
from .pydantic_models import HandLandmarksModel, ThumbUpCheckModel


def is_thumb_up(hand_landmarks) -> ThumbUpCheckModel:
    """
    Detect thumb up gesture - thumb extended upward, other fingers closed.

    Args:
        hand_landmarks: Hand landmarks detected using MediaPipe.

    Returns:
        ThumbUpCheckModel: Model indicating whether the thumb up gesture is detected.
    """
    logger.debug("is_thumb_up() called to detect thumb up gesture")

    try:
        # Get landmarks for all fingers
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Check if thumb is up (thumb tip is above thumb MCP)
        thumb_is_up = thumb_tip.y < thumb_mcp.y
        logger.debug(f"Thumb check: thumb_is_up={thumb_is_up}")

        # Check if other fingers are closed (finger tips are below their PIPs)
        index_closed = index_tip.y > index_pip.y
        middle_closed = middle_tip.y > middle_pip.y
        ring_closed = ring_tip.y > ring_pip.y
        pinky_closed = pinky_tip.y > pinky_pip.y

        logger.debug(
            f"Finger closure check: index_closed={index_closed}, middle_closed={middle_closed}, "
            f"ring_closed={ring_closed}, pinky_closed={pinky_closed}"
        )

        # Additional check that thumb is significantly extended
        thumb_extended = abs(thumb_tip.x - wrist.x) > 0.1
        logger.debug(f"Thumb extended check: thumb_extended={thumb_extended}")

        # Final condition to check if the thumb up gesture is detected
        is_thumb_up_detected = (
            thumb_is_up and thumb_extended and index_closed and middle_closed and ring_closed and pinky_closed
        )
        logger.info(f"Thumb up detected: {is_thumb_up_detected}")

        # Return the result in a ThumbUpCheckModel
        result = ThumbUpCheckModel(is_thumb_up=is_thumb_up_detected)
        logger.debug(f"Returning ThumbUpCheckModel: {result}")

        return result

    except Exception as e:
        logger.error(f"Error occurred in is_thumb_up(): {e}")
        raise
