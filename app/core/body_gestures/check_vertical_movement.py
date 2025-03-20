import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from loguru import logger

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setup the Pose function for videos.
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):
    """This function checks the posture (Jumping, Crouching or Standing) of the person in an image.
    
    Args:
        image:   The input image with a prominent person whose posture needs to be checked.
        results: The output of the pose landmarks detection on the input image.
        MID_Y:   The initial center y-coordinate of both shoulders landmarks of the person recorded during the start.
                 This helps determine the person's height when standing straight.
        draw:    If True, writes the posture on the output image.
        display: If True, displays the output image and returns nothing.

    Returns:
        output_image: The input image with the person's posture written, if specified.
        posture:      The detected posture (Jumping, Crouching, or Standing).

    """
    logger.debug("checkJumpCrouch() called. Processing frame...")

    # Get the height and width of the image.
    height, width, _ = image.shape
    logger.debug(f"Image dimensions: width={width}, height={height}, MID_Y={MID_Y}")

    # Create a copy of the input image to write the posture label on.
    output_image = image.copy()

    # Check if pose landmarks are detected
    if not results.pose_landmarks:
        logger.warning("No Pose Detected.")
        return output_image, "No Pose Detected"

    # Retrieve the y-coordinate of the left shoulder landmark.
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

    # Retrieve the y-coordinate of the right shoulder landmark.
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

    # Calculate the y-coordinate of the mid-point of both shoulders.
    actual_mid_y = abs(right_y + left_y) // 2
    logger.debug(f"Shoulder Y-coordinates -> Left: {left_y}, Right: {right_y}, MID_Y: {actual_mid_y}")

    # Calculate the upper and lower bounds of the threshold.
    lower_bound = MID_Y - 15
    upper_bound = MID_Y + 100

    # Determine posture based on threshold values
    if actual_mid_y < lower_bound:
        posture = "Jumping"
        logger.info("Posture detected: JUMPING")

    elif actual_mid_y > upper_bound:
        posture = "Crouching"
        logger.info("Posture detected: CROUCHING")

    else:
        posture = "Standing"
        logger.info("Posture detected: STANDING")

    # Check if the posture and a horizontal line at the threshold should be drawn.
    if draw:
        # Write the posture of the person on the image.
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        # Draw a line at the initial center y-coordinate of the person (threshold).
        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)
        logger.debug(f"Drew posture label '{posture}' and MID_Y reference line on image.")

    # Check if the output image should be displayed.
    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")
        logger.debug("Displaying processed output image.")

    else:
        logger.debug(f"Returning processed image with detected posture: {posture}")
        return output_image, posture
