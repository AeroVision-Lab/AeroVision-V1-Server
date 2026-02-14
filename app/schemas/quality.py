"""
Schemas for quality assessment API.
"""

from pydantic import BaseModel, Field

from app.schemas.common import Meta


class QualityDetails(BaseModel):
    """Detailed quality scores."""

    sharpness: float = Field(..., ge=0, le=1, description="Sharpness score (0-1)")
    exposure: float = Field(..., ge=0, le=1, description="Exposure score (0-1)")
    composition: float = Field(..., ge=0, le=1, description="Composition score (0-1)")
    noise: float = Field(..., ge=0, le=1, description="Noise score (0-1)")
    color: float = Field(..., ge=0, le=1, description="Color score (0-1)")


class QualityResult(BaseModel):
    """Quality assessment result."""

    pass_: bool = Field(..., alias="pass", description="Whether image passes quality threshold")
    score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")
    details: QualityDetails = Field(..., description="Detailed quality scores")


class QualityResponse(QualityResult):
    """Quality assessment response with metadata."""

    meta: Meta


class BatchQualityItem(BaseModel):
    """Single item in batch quality response."""

    index: int = Field(..., ge=0, description="Original image index in request")
    success: bool = Field(..., description="Whether this item was processed successfully")
    data: QualityResult | None = Field(None, description="Quality result if successful")
    error: str | None = Field(None, description="Error message if failed")


class BatchQualityResponse(BaseModel):
    """Batch quality assessment response."""

    total: int = Field(..., ge=0, description="Total number of images")
    successful: int = Field(..., ge=0, description="Number of successful assessments")
    failed: int = Field(..., ge=0, description="Number of failed assessments")
    results: list[BatchQualityItem] = Field(..., description="Individual results")
