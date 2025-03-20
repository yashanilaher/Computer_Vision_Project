import cv2
import mediapipe as mp
from math import hypot
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


def checkHandsJoined(image, results, draw=False, display=False):
    '''
    This function checks whether the hands of the person are joined or not in an image.
    Args:
        image:   The input image with a prominent person whose hands status (joined or not) needs to be classified.
        results: The output of the pose landmarks detection on the input image.
        draw:    A boolean value that is if set to true the function writes the hands status & distance on the output image. 
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image: The same input image but with the classified hands status written, if it was specified.
        hand_status:  The classified status of the hands whether they are joined or not.
    '''
    
    logger.debug("checkHandsJoined() called. Processing frame...")

    # Get the height and width of the input image.
    height, width, _ = image.shape
    logger.debug(f"Image dimensions: width={width}, height={height}")

    # Create a copy of the input image to write the hands status label on.
    output_image = image.copy()

    # Check if the pose landmarks are detected.
    if not results.pose_landmarks:
        logger.warning("No Pose Detected.")
        return output_image, "No Pose Detected"

    # Get the left wrist landmark x and y coordinates.
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    left_wrist_landmark = (int(left_wrist.x * width), int(left_wrist.y * height))

    # Get the right wrist landmark x and y coordinates.
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    right_wrist_landmark = (int(right_wrist.x * width), int(right_wrist.y * height))
    logger.debug(f"Left wrist: {left_wrist_landmark}, Right wrist: {right_wrist_landmark}")

    # Calculate the euclidean distance between the left and right wrist.
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                   left_wrist_landmark[1] - right_wrist_landmark[1]))
    logger.debug(f"Euclidean distance between wrists: {euclidean_distance}")

    # Get the shoulder width to normalize the distance.
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width = int(hypot(left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y) * width)
    logger.debug(f"Shoulder width: {shoulder_width}")

    # Normalize the distance by shoulder width.
    normalized_distance = euclidean_distance / shoulder_width
    logger.debug(f"Normalized distance: {normalized_distance:.2f}")

    # Compare the normalized distance with a threshold to check if both hands are joined.
    if normalized_distance < 0.5:  # Adjust this threshold as needed.
        # Set the hands status to joined.
        hand_status = 'Hands Joined'
        # Set the color value to green.
        color = (0, 255, 0)
        logger.info("Hands are joined.")
    else:
        # Set the hands status to not joined.
        hand_status = 'Hands Not Joined'
        # Set the color value to red.
        color = (0, 0, 255)
        logger.info("Hands are not joined.")

    # Check if the Hands Joined status and hands distance are specified to be written on the output image.
    if draw:
        # Write the classified hands status on the image. 
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
        # Write the distance between the wrists on the image. 
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        logger.debug(f"Drawing status on image: {hand_status}, Distance: {euclidean_distance}")

    # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
        logger.debug("Displaying output image with annotated information.")

    else:
        # Return the output image and the classified hands status indicating whether the hands are joined or not.
        logger.debug(f"Returning processed image and status: {hand_status}")
        return output_image, hand_status
