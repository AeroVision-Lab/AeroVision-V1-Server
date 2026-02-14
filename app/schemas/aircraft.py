"""
Schemas for aircraft type classification API.
"""

from pydantic import BaseModel, Field

from app.schemas.common import Meta


class Prediction(BaseModel):
    """Single prediction result."""

    class_: str = Field(..., alias="class", description="Predicted class name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class AircraftResult(BaseModel):
    """Aircraft classification result."""

    top1: Prediction = Field(..., description="Top-1 prediction")
    top_k: int = Field(..., ge=1, description="Number of top-k predictions returned")
    predictions: list[Prediction] = Field(..., description="Top-k predictions")


class AircraftResponse(AircraftResult):
    """Aircraft classification response with metadata."""

    meta: Meta


class BatchAircraftItem(BaseModel):
    """Single item in batch aircraft response."""

    index: int = Field(..., ge=0, description="Original image index in request")
    success: bool = Field(..., description="Whether this item was processed successfully")
    data: AircraftResult | None = Field(None, description="Aircraft result if successful")
    error: str | None = Field(None, description="Error message if failed")


class BatchAircraftResponse(BaseModel):
    """Batch aircraft classification response."""

    total: int = Field(..., ge=0, description="Total number of images")
    successful: int = Field(..., ge=0, description="Number of successful classifications")
    failed: int = Field(..., ge=0, description="Number of failed classifications")
    results: list[BatchAircraftItem] = Field(..., description="Individual results")
