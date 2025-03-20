# API Documentation

The **Gesture Control Game** project includes an API for advanced usage and integration. The API is built using **FastAPI** and **BentoML** and provides endpoints for gesture detection and game control.

## API Endpoints

### 1. **Process Frame**
- **Endpoint**: `/process_frame`
- **Method**: `POST`
- **Description**: Processes a video frame and returns gesture recognition results.
- **Request Body**:
    ```json
    {
      "image": "base64_encoded_image",
      "mid_y": 300,  // Optional: Reference Y-coordinate for vertical movement
      "game_started": false,  // Whether the game has started
      "gesture_type": "body"  // "body" or "hands"
    }
    ```
- **Response**:
    ```json
    {
      "success": true,
      "message": "Frame processed successfully",
      "data": {
        "frame": "base64_encoded_processed_frame",
        "results": {
          "action": "left",  // Detected action (e.g., left, right, up, down)
          "state": "neutral",  // Current state (e.g., neutral, left, right)
          "in_neutral": true,  // Whether the hand is in the neutral zone
          "thumb_up": false  // Whether a thumbs-up gesture was detected
        }
      }
    }
    ```

---

### 2. **Calculate Reference**
- **Endpoint**: `/calculate_reference`
- **Method**: `POST`
- **Description**: Calculates the reference Y-coordinate for vertical movement detection.
- **Request Body**:
    ```json
    {
      "image": "base64_encoded_image"
    }
    ```
- **Response**:
    ```json
    {
      "success": true,
      "message": "Reference Y-coordinate calculated successfully",
      "data": {
        "mid_y": 300  // Calculated reference Y-coordinate
      }
    }
    ```

---

## Running the API

To start the API server, run the following command:

```bash
python app/api/server.py