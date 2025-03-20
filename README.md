# Computer Vision Project

## Display Authorization Error Fix

If you encounter a display authorization error while running the project, use the following command to authorize it:

```bash
uv run xhost +local:
# Installation Guide

Follow these steps to set up the **Gesture Control Game** project on your local machine.

## Prerequisites

- Python 3.8 or higher
- Webcam
- Git
- Just

## Step 1: Clone the Repository

```bash
git clone https://github.com/yashanilaher/Computer_Vision_Project.git
cd Computer_Vision_Project/
```

## Step 2: Install Dependencies

Install the required Python libraries using the following command:

```bash
just setup
```

The required libraries include:

- `pyautogui`
- `opencv-python`
- `mediapipe`
- `matplotlib`

## Step 3: Run the Application  
To start the application, use one of the following commands based on your requirements:

#### 1. Start the FastAPI Server  
This command starts the FastAPI server for processing gestures via an API.  
```bash
just run-fastapi
```
- **Purpose**: Runs the `server.py` file located in `app/api/`.
- **Details**: Stops any existing instance of `server.py` before starting a new one.

#### 2. Start the Streamlit GUI  
This command starts the Streamlit GUI for the game interface.  
```bash
just run-gui
```
- **Purpose**: Runs the `main_window.py` file located in `app/gui/`.
- **Details**:
  - Stops any existing Streamlit instance running `main_window.py`.
  - Starts a new Streamlit instance for the game's graphical interface.

#### 3. Start the BentoML Server  
This command starts the BentoML server for serving the gesture recognition model.  
```bash
just serve-bm
```
- **Purpose**: Runs the `service.py` file located in `app/api/`.
- **Details**: Stops any existing instance of `service.py` before starting a new one.

#### 4. Start the Documentation Server  
This command starts the MkDocs server to view the project documentation locally.  
```bash
just run-mkdocs
```
- **Purpose**: Serves the documentation using the `mkdocs.yml` configuration file.
- **Details**: Stops any existing instance of `mkdocs serve` before starting a new one.

#### 5. Run Ruff Linter  
This command checks the codebase for linting errors using Ruff.  
```bash
just run-ruff
```
- **Purpose**: Ensures the code adheres to coding standards and best practices.
- **Details**: Runs `ruff check` on the entire project.

---

## Running in Docker or with GUI Issues

If you are running the script inside a Docker container, WSL, or as a different user and encounter GUI-related errors (e.g., `cv2.imshow()` not opening windows), run the following command before starting the script:

```bash
xhost +local:
```

This grants local access to the X server, which is required for displaying images in restricted environments.



## Explanation of Commands  
| Command | Description |
|---------|-------------|
| `just run-fastapi` | Starts the FastAPI server for gesture recognition. |
| `just run-gui` | Launches the Streamlit GUI for interacting with the game. |
| `just serve-bm` | Starts the BentoML server for serving the gesture recognition model. |
| `just run-mkdocs` | Serves the project documentation locally using MkDocs. |
| `just run-ruff` | Checks the codebase for linting errors using Ruff. |
