"""
Schemas for registration number OCR API.
"""

from pydantic import BaseModel, Field

from app.schemas.common import Meta


class YoloBox(BaseModel):
    """YOLO format bounding box with OCR result."""

    class_id: int = Field(..., ge=0, description="Class ID")
    x_center: float = Field(..., ge=0, le=1, description="Center X coordinate (normalized)")
    y_center: float = Field(..., ge=0, le=1, description="Center Y coordinate (normalized)")
    width: float = Field(..., ge=0, le=1, description="Box width (normalized)")
    height: float = Field(..., ge=0, le=1, description="Box height (normalized)")
    text: str = Field(..., description="Recognized text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class OcrMatch(BaseModel):
    """Single OCR match result."""

    text: str = Field(..., description="Recognized registration text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class RegistrationResult(BaseModel):
    """Registration number OCR result."""

    registration: str = Field(..., description="Recognized registration number")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence score (0-1)")
    raw_text: str = Field(..., description="Raw OCR text")
    all_matches: list[OcrMatch] = Field(default_factory=list, description="All OCR matches")
    yolo_boxes: list[YoloBox] = Field(default_factory=list, description="Bounding boxes in YOLO format")


class RegistrationResponse(RegistrationResult):
    """Registration OCR response with metadata."""

    meta: Meta


class BatchRegistrationItem(BaseModel):
    """Single item in batch registration response."""

    index: int = Field(..., ge=0, description="Original image index in request")
    success: bool = Field(..., description="Whether this item was processed successfully")
    data: RegistrationResult | None = Field(None, description="Registration result if successful")
    error: str | None = Field(None, description="Error message if failed")


class BatchRegistrationResponse(BaseModel):
    """Batch registration OCR response."""

    total: int = Field(..., ge=0, description="Total number of images")
    successful: int = Field(..., ge=0, description="Number of successful OCR")
    failed: int = Field(..., ge=0, description="Number of failed OCR")
    results: list[BatchRegistrationItem] = Field(..., description="Individual results")
