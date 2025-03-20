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


def detectPose(image, pose, draw=False, display=False):
    """This function performs the pose detection on the most prominent person in an image.

    Args:
        image:   The input image with a prominent person whose pose landmarks need to be detected.
        pose:    The pose function required to perform the pose detection.
        draw:    If True, the function draws pose landmarks on the output image.
        display: If True, the function displays the original and processed images and returns nothing.

    Returns:
        output_image: The input image with the detected pose landmarks drawn if specified.
        results:      The output of the pose landmarks detection on the input image.

    """
    logger.debug("detectPose() called. Processing frame...")

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.debug("Converted image to RGB format.")

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Check if any landmarks are detected
    if results.pose_landmarks:
        logger.info("Pose landmarks detected.")

        # Draw Pose Landmarks on the output image if requested
        if draw:
            mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=3, circle_radius=3),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                   thickness=2, circle_radius=2))
            logger.debug("Pose landmarks drawn on output image.")

    else:
        logger.warning("No pose landmarks detected.")

    # Check if the original and output images should be displayed.
    if display:
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")

        logger.debug("Displaying original and processed images.")

    else:
        logger.debug("Returning processed image and pose results.")
        return output_image, results
