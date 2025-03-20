"""Locust file to load test the game API with various endpoints."""

from pathlib import Path

from locust import HttpUser, between, task


# Custom exception for file not found errors
class CustomFileNotFoundError(Exception):
    """Exception raised when an expected file is not found."""

class QuickstartUser(HttpUser):
    """Load testing user for the game API with various endpoints."""

    wait_time = between(1, 2)
    host = "http://0.0.0.0:8000"

    @task(3)
    def hello_world(self) -> None:
        """Send a GET request to the root endpoint."""
        self.client.get("/")

    @task(4)
    def process_frame1(self) -> None:
        """Process a sample body gesture frame."""
        image_path = Path("test_image/body.png")

        if not image_path.exists():
            error_msg = f"Image file not found: {image_path}"
            raise CustomFileNotFoundError(error_msg)

        with image_path.open("rb") as image_file:
            self.client.post(
                "/process_frame",
                files={"file": image_file},
                data={"game_started": "true", "gesture_type": "body"},
            )

    @task(4)
    def process_frame2(self) -> None:
        """Process a sample hand gesture frame."""
        image_path = Path("test_image/hand.jpg")

        if not image_path.exists():
            error_msg = f"Image file not found: {image_path}"
            raise CustomFileNotFoundError(error_msg)

        with image_path.open("rb") as image_file:
            self.client.post(
                "/process_frame",
                files={"file": image_file},
                data={"game_started": "true", "gesture_type": "hand"},
            )

    @task(2)
    def calculate_reference(self) -> None:
        """Send a reference calculation request with a sample image."""
        image_path = Path("test_image/body.png")

        if not image_path.exists():
            error_msg = f"Image file not found: {image_path}"
            raise CustomFileNotFoundError(error_msg)

        with image_path.open("rb") as image_file:
            self.client.post(
                "/calculate_reference",
                files={"file": image_file},
            )
