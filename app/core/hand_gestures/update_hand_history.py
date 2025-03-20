from loguru import logger
from utils.config_types import LoggingConfigs
from utils.logging_client import setup_network_logger_client

from . import hand_positions_data, position_history_length

def update_hand_history(hand_x: float, hand_y: float) -> None:
    """Update the history of hand positions"""
    
    logger.debug(f"Updating hand position history: ({hand_x}, {hand_y})")
    
    hand_positions_data.hand_positions.append((hand_x, hand_y))
    
    if len(hand_positions_data.hand_positions) > position_history_length:
        removed_position = hand_positions_data.hand_positions.pop(0)
        logger.debug(f"Removed oldest hand position: {removed_position} (History length exceeded)")
    
    logger.info(f"Current hand position history length: {len(hand_positions_data.hand_positions)}")
