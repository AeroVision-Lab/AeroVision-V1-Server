"""
Schemas for aggregated review API.
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.aircraft import AircraftResult
from app.schemas.airline import AirlineResult
from app.schemas.common import Meta
from app.schemas.quality import QualityDetails
from app.schemas.registration import RegistrationResult


class ReviewQualityResult(BaseModel):
    """Quality result in review response."""

    score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")
    pass_: bool = Field(..., alias="pass", description="Whether image passes quality threshold")
    details: Optional[QualityDetails] = Field(None, description="Detailed quality scores")


class ReviewAircraftResult(BaseModel):
    """Aircraft result in review response."""

    type_code: str = Field(..., description="Aircraft type code")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class ReviewAirlineResult(BaseModel):
    """Airline result in review response."""

    airline_code: str = Field(..., description="Airline code")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class ReviewRegistrationResult(BaseModel):
    """Registration result in review response."""

    registration: str = Field(..., description="Recognized registration number")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    clarity: float = Field(..., ge=0, le=1, description="Registration clarity score (0-1)")


class ReviewResult(BaseModel):
    """Complete review result."""

    quality: ReviewQualityResult = Field(..., description="Quality assessment result")
    aircraft: ReviewAircraftResult = Field(..., description="Aircraft classification result")
    airline: Optional[ReviewAirlineResult] = Field(None, description="Airline classification result")
    registration: Optional[ReviewRegistrationResult] = Field(None, description="Registration OCR result")


class ReviewResponse(ReviewResult):
    """Review response with metadata."""

    meta: Meta


class BatchReviewItem(BaseModel):
    """Single item in batch review response."""

    index: int = Field(..., ge=0, description="Original image index in request")
    success: bool = Field(..., description="Whether this item was processed successfully")
    data: ReviewResult | None = Field(None, description="Review result if successful")
    error: str | None = Field(None, description="Error message if failed")


class BatchReviewResponse(BaseModel):
    """Batch review response."""

    total: int = Field(..., ge=0, description="Total number of images")
    successful: int = Field(..., ge=0, description="Number of successful reviews")
    failed: int = Field(..., ge=0, description="Number of failed reviews")
    results: list[BatchReviewItem] = Field(..., description="Individual results")
