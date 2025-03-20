import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from loguru import logger
from utils.config_types import LoggingConfigs
from utils.logging_client import setup_network_logger_client

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setup the Pose function for videos.
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


def checkLeftRight(image, results, draw=False, display=False):
    '''
    This function finds the horizontal position (left, center, right) of the person in an image.
    Args:
        image:   The input image with a prominent person whose the horizontal position needs to be found.
        results: The output of the pose landmarks detection on the input image.
        draw:    A boolean value that is if set to true the function writes the horizontal position on the output image. 
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:         The same input image but with the horizontal position written, if it was specified.
        horizontal_position:  The horizontal position (left, center, right) of the person in the input image.
    '''

    logger.debug("checkLeftRight() called. Processing frame...")

    # Declare a variable to store the horizontal position (left, center, right) of the person.
    horizontal_position = None

    # Get the height and width of the image.
    height, width, _ = image.shape
    logger.debug(f"Image dimensions: width={width}, height={height}")

    # Create a copy of the input image to write the horizontal position on.
    output_image = image.copy()

    # Check if pose landmarks are detected
    if not results.pose_landmarks:
        logger.warning("No Pose Detected.")
        return output_image, "No Pose Detected"

    # Retrieve the x-coordinate of the left shoulder landmark.
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

    # Retrieve the x-coordinate of the right shoulder landmark.
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    logger.debug(f"Left Shoulder X: {left_x}, Right Shoulder X: {right_x}")

    # Check if the person is at left that is when both shoulder landmarks x-coordinates
    # are less than or equal to the x-coordinate of the center of the image.
    if (right_x <= width // 2 and left_x <= width // 2):
        # Set the person's position to left.
        horizontal_position = 'Left'
        logger.info("Person detected on the LEFT side.")
    
    # Check if the person is at right that is when both shoulder landmarks x-coordinates
    # are greater than or equal to the x-coordinate of the center of the image.
    elif (right_x >= width // 2 and left_x >= width // 2):
        # Set the person's position to right.
        horizontal_position = 'Right'
        logger.info("Person detected on the RIGHT side.")
    
    # Check if the person is at center that is when right shoulder landmark x-coordinate is greater than or equal to
    # and left shoulder landmark x-coordinate is less than or equal to the x-coordinate of the center of the image.
    elif (right_x >= width // 2 and left_x <= width // 2):
        # Set the person's position to center.
        horizontal_position = 'Center'
        logger.info("Person detected in the CENTER.")
    
    # Check if the person's horizontal position and a line at the center of the image is specified to be drawn.
    if draw:
        # Write the horizontal position of the person on the image. 
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        
        # Draw a line at the center of the image.
        cv2.line(output_image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
        logger.debug(f"Drew position and center line on image: {horizontal_position}")

    # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
        logger.debug("Displaying output image with position information.")
    else:
        # Return the output image and the person's horizontal position.
        logger.debug(f"Returning processed image with horizontal position: {horizontal_position}")
        return output_image, horizontal_position
