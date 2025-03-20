from loguru import logger
from utils.config_types import LoggingConfigs
from utils.logging_client import setup_network_logger_client
from . import neutral_zone_x, neutral_zone_y, neutral_zone_size
from .pydantic_models import HandPositionModel, NeutralZoneCheckModel


def is_in_neutral_zone(hand_x: float, hand_y: float) -> NeutralZoneCheckModel:
    """
    Check if the hand is in the neutral zone (center of the frame).
    
    Args:
        hand_x (float): X-coordinate of the hand.
        hand_y (float): Y-coordinate of the hand.

    Returns:
        NeutralZoneCheckModel: Model indicating whether the hand is in the neutral zone.
    """
    logger.debug(f"is_in_neutral_zone() called with hand_x={hand_x}, hand_y={hand_y}")
    
    try:
        # Create a HandPositionModel to hold hand coordinates
        position = HandPositionModel(hand_x=hand_x, hand_y=hand_y)
        logger.debug(f"Hand position recorded: {position}")

        # Check if the hand is within the neutral zone
        is_neutral = (
            abs(position.hand_x - neutral_zone_x) < neutral_zone_size and
            abs(position.hand_y - neutral_zone_y) < neutral_zone_size
        )
        
        logger.info(f"Neutral zone check: {'IN' if is_neutral else 'OUT'} of the neutral zone.")
        
        # Return the result wrapped in NeutralZoneCheckModel
        result = NeutralZoneCheckModel(neutral_zone_check=is_neutral)
        logger.debug(f"Returning NeutralZoneCheckModel: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error occurred in is_in_neutral_zone(): {e}")
        raise
