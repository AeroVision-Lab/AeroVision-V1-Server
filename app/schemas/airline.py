"""
Schemas for airline classification API.
"""

from pydantic import BaseModel, Field

from app.schemas.common import Meta


class Prediction(BaseModel):
    """Single prediction result."""

    class_: str = Field(..., alias="class", description="Predicted airline code")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class AirlineResult(BaseModel):
    """Airline classification result."""

    top1: Prediction = Field(..., description="Top-1 prediction")
    top_k: int = Field(..., ge=1, description="Number of top-k predictions returned")
    predictions: list[Prediction] = Field(..., description="Top-k predictions")


class AirlineResponse(AirlineResult):
    """Airline classification response with metadata."""

    meta: Meta


class BatchAirlineItem(BaseModel):
    """Single item in batch airline response."""

    index: int = Field(..., ge=0, description="Original image index in request")
    success: bool = Field(..., description="Whether this item was processed successfully")
    data: AirlineResult | None = Field(None, description="Airline result if successful")
    error: str | None = Field(None, description="Error message if failed")


class BatchAirlineResponse(BaseModel):
    """Batch airline classification response."""

    total: int = Field(..., ge=0, description="Total number of images")
    successful: int = Field(..., ge=0, description="Number of successful classifications")
    failed: int = Field(..., ge=0, description="Number of failed classifications")
    results: list[BatchAirlineItem] = Field(..., description="Individual results")
