"""
Schemas for quality assessment API.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Meta


class QualityDetails(BaseModel):
    """Detailed quality scores."""

    sharpness: float = Field(..., ge=0, le=1, description="Sharpness score (0-1)")
    exposure: float = Field(..., ge=0, le=1, description="Exposure score (0-1)")
    composition: float = Field(..., ge=0, le=1, description="Composition score (0-1)")
    noise: float = Field(..., ge=0, le=1, description="Noise score (0-1)")
    color: float = Field(..., ge=0, le=1, description="Color score (0-1)")


class RuleViolation(BaseModel):
    """Single rule violation information."""

    rule_id: str = Field(..., description="Rule ID, e.g., Rules1.1.1")
    rule_name: str = Field(..., description="Rule name, e.g., '模糊/虚焦'")
    severity: Literal["critical", "major", "minor"] = Field(..., description="Violation severity")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    description: str = Field(..., description="Detailed problem description")
    source: Literal["opencv", "qwen"] = Field(..., description="Detection source")


class QualityResult(BaseModel):
    """Quality assessment result."""

    pass_: bool = Field(..., alias="pass", description="Whether image passes quality threshold")
    score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")

    # Core: violations list
    violations: list[RuleViolation] = Field(default_factory=list, description="List of rule violations")

    # Optional: dimension scores (backward compatibility)
    details: QualityDetails | None = Field(None, description="Detailed quality scores")

    # Optional: improvement suggestions
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")


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
