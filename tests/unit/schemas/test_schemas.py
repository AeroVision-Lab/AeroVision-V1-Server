"""
Unit tests for Pydantic schemas.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.schemas.common import (
    ImageInput,
    BatchImageInput,
    Meta,
    SuccessResponse,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    StatsResponse,
)
from app.schemas.quality import (
    QualityDetails,
    QualityResult,
    QualityResponse,
    BatchQualityItem,
    BatchQualityResponse,
)
from app.schemas.aircraft import (
    AircraftResponse,
    AircraftResult,
    BatchAircraftItem,
    BatchAircraftResponse,
    Prediction,
)
from app.schemas.airline import (
    AirlineResponse,
    AirlineResult,
    BatchAirlineItem,
    BatchAirlineResponse,
)
from app.schemas.registration import (
    RegistrationResponse,
    RegistrationResult,
    BatchRegistrationItem,
    BatchRegistrationResponse,
    YoloBox,
    OcrMatch,
)
from app.schemas.review import (
    ReviewResponse,
    ReviewResult,
    BatchReviewResponse,
    BatchReviewItem,
    ReviewQualityResult,
    ReviewAircraftResult,
    ReviewAirlineResult,
    ReviewRegistrationResult,
)


class TestCommonSchemas:
    """Tests for common schemas."""

    def test_image_input_valid(self):
        """Test valid image input."""
        data = {"image": "data:image/jpeg;base64,/9j/4AAQ"}
        result = ImageInput(**data)
        assert result.image == "data:image/jpeg;base64,/9j/4AAQ"

    def test_batch_image_input_valid(self):
        """Test valid batch image input."""
        data = {"images": ["img1", "img2"]}
        result = BatchImageInput(**data)
        assert len(result.images) == 2

    def test_batch_image_input_empty_fails(self):
        """Test empty batch image input fails."""
        with pytest.raises(ValidationError):
            BatchImageInput(images=[])

    def test_batch_image_input_too_many_fails(self):
        """Test batch with too many images fails."""
        with pytest.raises(ValidationError):
            BatchImageInput(images=["img"] * 51)

    def test_meta_creation(self):
        """Test metadata creation."""
        meta = Meta(processing_time_ms=123.45)
        assert meta.processing_time_ms == 123.45
        assert isinstance(meta.timestamp, datetime)

    def test_error_response(self):
        """Test error response."""
        error = ErrorDetail(code="TEST_ERROR", message="Test error message")
        response = ErrorResponse(error=error)
        assert response.success is False
        assert response.error.code == "TEST_ERROR"

    def test_health_response(self):
        """Test health response."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0
        )
        assert response.status == "healthy"
        assert response.version == "1.0.0"

    def test_stats_response(self):
        """Test stats response."""
        response = StatsResponse(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            uptime_seconds=3600.0,
            requests_per_second=0.028
        )
        assert response.total_requests == 100
        assert response.successful_requests == 95


class TestQualitySchemas:
    """Tests for quality schemas."""

    def test_quality_details_valid(self):
        """Test valid quality details."""
        data = {
            "sharpness": 0.9,
            "exposure": 0.8,
            "composition": 0.85,
            "noise": 0.88,
            "color": 0.82
        }
        result = QualityDetails(**data)
        assert result.sharpness == 0.9

    def test_quality_details_out_of_range_fails(self):
        """Test quality details with out of range values fails."""
        with pytest.raises(ValidationError):
            QualityDetails(sharpness=1.5)

    def test_quality_result_valid(self):
        """Test valid quality result."""
        details = QualityDetails(
            sharpness=0.9, exposure=0.8, composition=0.85, noise=0.88, color=0.82
        )
        result = QualityResult(**{"pass": True, "score": 0.85, "details": details})
        assert result.pass_ is True
        assert result.score == 0.85

    def test_batch_quality_item_valid(self):
        """Test valid batch quality item."""
        details = QualityDetails(
            sharpness=0.9, exposure=0.8, composition=0.85, noise=0.88, color=0.82
        )
        data = {
            "index": 0,
            "success": True,
            "data": {
                "pass": True,
                "score": 0.85,
                "details": details.model_dump()
            }
        }
        result = BatchQualityItem(**data)
        assert result.index == 0
        assert result.success is True


class TestAircraftSchemas:
    """Tests for aircraft schemas."""

    def test_prediction_valid(self):
        """Test valid prediction."""
        data = {"class": "A320", "confidence": 0.85}
        result = Prediction(**data)
        assert result.class_ == "A320"
        assert result.confidence == 0.85

    def test_prediction_confidence_out_of_range_fails(self):
        """Test prediction with invalid confidence fails."""
        with pytest.raises(ValidationError):
            Prediction(**{"class": "A320", "confidence": 1.5})

    def test_aircraft_result_valid(self):
        """Test valid aircraft result."""
        predictions = [
            {"class": "A320", "confidence": 0.85},
            {"class": "B738", "confidence": 0.10}
        ]
        data = {
            "top1": {"class": "A320", "confidence": 0.85},
            "top_k": 5,
            "predictions": predictions
        }
        result = AircraftResult(**data)
        assert result.top1.class_ == "A320"
        assert len(result.predictions) == 2


class TestAirlineSchemas:
    """Tests for airline schemas."""

    def test_airline_result_valid(self):
        """Test valid airline result."""
        predictions = [
            {"class": "CEA", "confidence": 0.75},
            {"class": "CSN", "confidence": 0.15}
        ]
        data = {
            "top1": {"class": "CEA", "confidence": 0.75},
            "top_k": 5,
            "predictions": predictions
        }
        result = AirlineResult(**data)
        assert result.top1.class_ == "CEA"


class TestRegistrationSchemas:
    """Tests for registration schemas."""

    def test_yolo_box_valid(self):
        """Test valid YOLO box."""
        data = {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.3,
            "width": 0.2,
            "height": 0.1,
            "text": "B-1234",
            "confidence": 0.95
        }
        result = YoloBox(**data)
        assert result.text == "B-1234"

    def test_yolo_box_out_of_range_fails(self):
        """Test YOLO box with out of range values fails."""
        with pytest.raises(ValidationError):
            YoloBox(
                class_id=0,
                x_center=1.5,
                y_center=0.3,
                width=0.2,
                height=0.1,
                text="B-1234",
                confidence=0.95
            )

    def test_registration_result_valid(self):
        """Test valid registration result."""
        data = {
            "registration": "B-1234",
            "confidence": 0.95,
            "raw_text": "B-1234"
        }
        result = RegistrationResult(**data)
        assert result.registration == "B-1234"


class TestReviewSchemas:
    """Tests for review schemas."""

    def test_review_quality_result_valid(self):
        """Test valid review quality result."""
        data = {
            "score": 0.85,
            "pass": True,
            "details": {
                "sharpness": 0.9,
                "exposure": 0.8,
                "composition": 0.85,
                "noise": 0.88,
                "color": 0.82
            }
        }
        result = ReviewQualityResult(**data)
        assert result.score == 0.85
        assert result.pass_ is True

    def test_review_aircraft_result_valid(self):
        """Test valid review aircraft result."""
        data = {
            "type_code": "A320",
            "confidence": 0.85
        }
        result = ReviewAircraftResult(**data)
        assert result.type_code == "A320"

    def test_review_result_valid(self):
        """Test valid complete review result."""
        data = {
            "quality": {
                "score": 0.85,
                "pass": True,
                "details": {
                    "sharpness": 0.9,
                    "exposure": 0.8,
                    "composition": 0.85,
                    "noise": 0.88,
                    "color": 0.82
                }
            },
            "aircraft": {
                "type_code": "A320",
                "confidence": 0.85
            }
        }
        result = ReviewResult(**data)
        assert result.quality.score == 0.85
        assert result.aircraft.type_code == "A320"
