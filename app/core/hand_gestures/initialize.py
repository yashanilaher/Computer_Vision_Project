import mediapipe as mp
from loguru import logger

from .pydantic_models import HandsOutputModel


def initialize_hands() -> HandsOutputModel:
    """Initialize the Mediapipe Hands module with specified parameters.

    Returns:
        HandsOutputModel: An object containing the initialized hands module and drawing utilities.

    """
    logger.debug("initialize_hands() called. Initializing Mediapipe Hands...")

    try:
        # Initialize Mediapipe Hands module
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        logger.info("Mediapipe Hands module initialized successfully.")

        # Initialize Mediapipe drawing utilities
        mp_drawing = mp.solutions.drawing_utils
        logger.info("Mediapipe Drawing utilities initialized successfully.")

        # Return HandsOutputModel with initialized modules
        logger.debug("Returning HandsOutputModel with initialized modules.")
        return HandsOutputModel(hands=hands, mp_drawing=mp_drawing)

    except Exception as e:
        logger.error(f"Error initializing Mediapipe Hands: {e}")
        raise
