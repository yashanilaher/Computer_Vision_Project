from loguru import logger

from . import hand_positions_data
from .pydantic_models import MovementVectorModel


def calculate_movement_vector() -> MovementVectorModel:
    """Calculate the movement vector from hand position history"""
    logger.debug("calculate_movement_vector() called.")

    # Check if there are enough hand positions to calculate movement
    if len(hand_positions_data.hand_positions) < 2:
        logger.warning("Not enough hand position data. Returning default (0,0).")
        return MovementVectorModel(x=0, y=0)

    # Get the latest and initial hand positions
    current_x, current_y = hand_positions_data.hand_positions[-1]
    past_x, past_y = hand_positions_data.hand_positions[0]

    # Compute movement vector
    movement_vector = MovementVectorModel(x=current_x - past_x, y=current_y - past_y)

    logger.info(f"Movement vector calculated: x={movement_vector.x}, y={movement_vector.y}")

    return movement_vector
