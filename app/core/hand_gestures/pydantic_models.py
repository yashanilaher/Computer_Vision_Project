
from pydantic import BaseModel, Field


class HandPositionsModel(BaseModel):
    hand_positions: list[tuple[float, float]] = Field(...)

class MovementVectorModel(BaseModel):
    x: float
    y: float

class HandsOutputModel(BaseModel):
    hands: object  # MediaPipe Hands object
    mp_drawing: object  # MediaPipe Drawing Utils object

class HandPositionModel(BaseModel):
    hand_x: float
    hand_y: float

class NeutralZoneCheckModel(BaseModel):
    neutral_zone_check: bool

class LandmarkModel(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    z: float = Field(...)

class HandLandmarksModel(BaseModel):
    landmark: dict[int, LandmarkModel]

class ThumbUpCheckModel(BaseModel):
    is_thumb_up: bool

class GestureDetectionResult(BaseModel):
    hand_x: float
    hand_y: float
    in_neutral: bool
    state: str
    thumb_up: bool
    action: str | None = None
    dx: float | None = None
    dy: float | None = None
