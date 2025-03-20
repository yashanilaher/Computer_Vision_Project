from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class GestureResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Message describing the result")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")

class FrameRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    mid_y: Optional[int] = Field(None, description="Reference Y-coordinate for vertical movement detection")
    game_started: bool = Field(False, description="Whether the game has started")
    
class ReferenceRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    
class GestureConfig(BaseModel):
    min_detection_confidence: float = Field(0.7, description="Minimum confidence for detection")
    min_tracking_confidence: float = Field(0.7, description="Minimum confidence for tracking")
    model_complexity: int = Field(1, description="Model complexity (0, 1, or 2)")
    static_image_mode: bool = Field(False, description="Whether to process static images")
    
class ApiConfig(BaseModel):
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    prefix: str = Field("", description="API prefix")
    enable_cors: bool = Field(True, description="Enable CORS")
    allowed_origins: List[str] = Field(["*"], description="Allowed origins for CORS")
    
class Config(BaseModel):
    gesture: GestureConfig = Field(default_factory=GestureConfig, description="Gesture recognition configuration")
    api: ApiConfig = Field(default_factory=ApiConfig, description="API configuration")