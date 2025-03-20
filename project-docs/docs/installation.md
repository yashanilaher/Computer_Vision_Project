# Installation Guide

Follow these steps to set up the **Gesture Control Game** project on your local machine.

## Prerequisites

- Python 3.8 or higher
- Webcam
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/gesture-control-game.git
cd gesture-control-game
```

## Step 2: Install Dependencies

Install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

The required libraries include:

- `pyautogui`
- `opencv-python`
- `mediapipe`
- `matplotlib`

## Step 3: Run the Application

To start the application, run the following command:

```bash
python main.py
```

## Step 4: Play the Game

1. Open Poki Games and choose **Subway Surfers** or **Temple Run**.
2. Use the following gestures to control the game:

   - **Start/Resume Game:** Use body posture (join hands) or hand gestures (thumbs up).
   - **Move Left/Right:** Use body posture (lean left/right) or hand gestures (move hand left/right).
   - **Jump:** Use body posture (stand on tiptoes) or hand gestures (move hand up).
   - **Crouch/Slide:** Use body posture (crouch down) or hand gestures (move hand down).

## Running in Docker or with GUI Issues

If you are running the script inside a Docker container, WSL, or as a different user and encounter GUI-related errors (e.g., `cv2.imshow()` not opening windows), run the following command before starting the script:

```bash
xhost +local:
```

This grants local access to the X server, which is required for displaying images in restricted environments.

## Next Steps

- **[Usage Guide](#)**
- **[API Documentation](#)**