"""
Common Pydantic schemas for Aerovision-V1-Server.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    """Input schema for a single image."""

    image: str = Field(
        ...,
        description="Image as base64 encoded string or URL",
        examples=["data:image/jpeg;base64,/9j/4AAQ..."]
    )


class BatchImageInput(BaseModel):
    """Input schema for multiple images."""

    images: list[str] = Field(
        ...,
        description="List of images as base64 encoded strings or URLs",
        min_length=1,
        max_length=50
    )


class Meta(BaseModel):
    """Metadata included in all responses."""

    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds",
        ge=0
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp in UTC"
    )


class SuccessResponse(BaseModel):
    """Base success response wrapper."""

    success: bool = Field(default=True, description="Request success status")
    data: Any = Field(..., description="Response data")
    meta: Meta = Field(..., description="Response metadata")


class ErrorDetail(BaseModel):
    """Error detail information."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(default=None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    success: bool = Field(default=False, description="Request success status")
    error: ErrorDetail = Field(..., description="Error information")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(default=False, description="Whether inference models are loaded")
    gpu_available: bool = Field(default=False, description="Whether GPU is available")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class StatsResponse(BaseModel):
    """Request statistics response."""

    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    requests_per_second: float = Field(..., description="Average requests per second")
