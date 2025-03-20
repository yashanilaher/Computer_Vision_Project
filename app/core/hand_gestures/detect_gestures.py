import time

import pyautogui
from loguru import logger

from . import (
    cooldown,
    current_state,
    down_threshold,
    hand_positions_data,
    in_neutral_zone,
    jump_threshold,
    last_action_time,
    left_threshold,
    mp_hands,
    reference_hand_x,
    reference_hand_y,
    right_threshold,
    state_start_time,
    thumb_up_cooldown,
)
from .is_in_neutral_zone import is_in_neutral_zone
from .is_thumb_up import is_thumb_up
from .pydantic_models import GestureDetectionResult
from .update_hand_history import update_hand_history


def detect_gestures(hand_landmarks) -> GestureDetectionResult:
    """Detect gestures based on hand landmarks and perform corresponding actions.

    Args:
        hand_landmarks: The landmarks detected from the hand.

    Returns:
        GestureDetectionResult: Result containing debug info and detected actions.

    """
    logger.debug("detect_gestures() called. Processing hand landmarks...")

    # Extract key points from hand landmarks
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Calculate hand center position
    hand_x = wrist.x
    hand_y = wrist.y
    logger.info(f"Hand position detected: x={hand_x}, y={hand_y}")

    # Update hand position history
    update_hand_history(hand_x, hand_y)
    logger.debug("Hand position history updated.")

    # Check if the hand is in the neutral zone
    now_in_neutral = is_in_neutral_zone(hand_x, hand_y).neutral_zone_check
    logger.info(f"Hand in neutral zone: {now_in_neutral}")

    # Debug info initialization
    debug_info = GestureDetectionResult(
        hand_x=hand_x,
        hand_y=hand_y,
        in_neutral=now_in_neutral,
        state=current_state[0],
        thumb_up=False,
        action=None,
        dx=None,
        dy=None,
    )

    # State machine logic
    current_time = time.time()

    # Check for thumb-up gesture
    thumb_up_detected = is_thumb_up(hand_landmarks).is_thumb_up
    debug_info.thumb_up = thumb_up_detected
    last_thumb_up_time = 0

    if thumb_up_detected and current_time - last_thumb_up_time > thumb_up_cooldown[0]:
        logger.info("Thumb-up gesture detected. Pressing 'space'.")
        pyautogui.press("space")
        current_state[0] = "thumb_up"
        last_thumb_up_time = current_time
        last_action_time[0] = current_time
        debug_info.action = "space"
        return debug_info

    # When entering the neutral zone, store the reference position
    if now_in_neutral and not in_neutral_zone[0]:
        reference_hand_x[0] = hand_x
        reference_hand_y[0] = hand_y
        hand_positions_data.hand_positions.clear()
        logger.info("Neutral zone entered. Reference position stored.")

    # Update neutral zone status
    in_neutral_zone[0] = now_in_neutral

    # Reset to neutral state if in non-neutral state and back in neutral zone
    if current_state[0] != "neutral" and now_in_neutral:
        current_state[0] = "neutral"
        state_start_time[0] = current_time
        logger.info("Resetting to neutral state.")
        return debug_info

    # Detect gestures if outside neutral zone and no recent actions
    if (
        current_state[0] == "neutral"
        and not now_in_neutral
        and current_time - last_action_time[0] > cooldown[0]
    ):
        # Calculate movement from reference position
        if reference_hand_x[0] is not None and reference_hand_y[0] is not None:
            dx = hand_x - reference_hand_x[0]
            dy = hand_y - reference_hand_y[0]

            debug_info.dx = dx
            debug_info.dy = dy

            logger.info(f"Movement vector calculated: dx={dx}, dy={dy}")

            # Check for deliberate left movement
            if dx < -left_threshold and abs(dy) < down_threshold:
                logger.info("Left movement detected. Pressing 'left'.")
                pyautogui.press("left")
                current_state[0] = "left"
                last_action_time[0] = current_time
                debug_info.action = "left"

            # Check for deliberate right movement
            elif dx > right_threshold and abs(dy) < down_threshold:
                logger.info("Right movement detected. Pressing 'right'.")
                pyautogui.press("right")
                current_state[0] = "right"
                last_action_time[0] = current_time
                debug_info.action = "right"

            # Check for downward (crouch/slide) movement
            elif dy > down_threshold and abs(dx) < right_threshold:
                logger.info("Downward movement detected. Pressing 'down'.")
                pyautogui.press("down")
                current_state[0] = "down"
                last_action_time[0] = current_time
                debug_info.action = "down"

            # Check for jump gesture (upward motion)
            elif dy < -jump_threshold and abs(dx) < right_threshold:
                logger.info("Jump detected (upward motion). Pressing 'up'.")
                pyautogui.press("up")
                current_state[0] = "jump"
                last_action_time[0] = current_time
                debug_info.action = "jump"

    logger.debug("Returning gesture detection results.")
    return debug_info
