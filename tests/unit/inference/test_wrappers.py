"""
Unit tests for inference wrappers.
"""

import pytest

from app.inference.wrappers import (
    wrap_quality_result,
    wrap_aircraft_result,
    wrap_airline_result,
    wrap_registration_result,
)
from app.schemas.quality import QualityResult
from app.schemas.aircraft import AircraftResult
from app.schemas.airline import AirlineResult
from app.schemas.registration import RegistrationResult


class TestWrapQualityResult:
    """Tests for wrap_quality_result."""

    def test_wrap_full_quality_result(self):
        """Test wrapping a full quality result."""
        inference_result = {
            "success": True,
            "pass": True,
            "score": 0.85,
            "details": {
                "sharpness": 0.90,
                "exposure": 0.80,
                "composition": 0.85,
                "noise": 0.88,
                "color": 0.82
            }
        }

        result = wrap_quality_result(inference_result)

        assert isinstance(result, QualityResult)
        assert result.pass_ is True
        assert result.score == 0.85
        assert result.details.sharpness == 0.90
        assert result.details.exposure == 0.80
        assert result.details.composition == 0.85
        assert result.details.noise == 0.88
        assert result.details.color == 0.82

    def test_wrap_minimal_quality_result(self):
        """Test wrapping a minimal quality result."""
        inference_result = {"pass": False, "score": 0.3}

        result = wrap_quality_result(inference_result)

        assert result.pass_ is False
        assert result.score == 0.3
        assert result.details.sharpness == 0.0
        assert result.details.exposure == 0.0


class TestWrapAircraftResult:
    """Tests for wrap_aircraft_result."""

    def test_wrap_full_aircraft_result(self):
        """Test wrapping a full aircraft result."""
        inference_result = {
            "predictions": [
                {"class": "A320", "confidence": 0.85},
                {"class": "B738", "confidence": 0.10},
                {"class": "A321", "confidence": 0.05}
            ],
            "top1": {"class": "A320", "confidence": 0.85},
            "top_k": 5
        }

        result = wrap_aircraft_result(inference_result)

        assert isinstance(result, AircraftResult)
        assert result.top1.class_ == "A320"
        assert result.top1.confidence == 0.85
        assert result.top_k == 5
        assert len(result.predictions) == 3
        assert result.predictions[0].class_ == "A320"

    def test_wrap_empty_aircraft_result(self):
        """Test wrapping an empty aircraft result."""
        inference_result = {
            "predictions": [],
            "top1": {},
            "top_k": 1  # Minimal valid top_k value
        }

        result = wrap_aircraft_result(inference_result)

        assert result.top1.class_ == ""
        assert result.top1.confidence == 0.0
        assert len(result.predictions) == 0
        assert result.top_k == 1


class TestWrapAirlineResult:
    """Tests for wrap_airline_result."""

    def test_wrap_full_airline_result(self):
        """Test wrapping a full airline result."""
        inference_result = {
            "predictions": [
                {"class": "CEA", "confidence": 0.75},
                {"class": "CSN", "confidence": 0.15}
            ],
            "top1": {"class": "CEA", "confidence": 0.75},
            "top_k": 5
        }

        result = wrap_airline_result(inference_result)

        assert isinstance(result, AirlineResult)
        assert result.top1.class_ == "CEA"
        assert result.top1.confidence == 0.75
        assert len(result.predictions) == 2


class TestWrapRegistrationResult:
    """Tests for wrap_registration_result."""

    def test_wrap_full_registration_result(self):
        """Test wrapping a full registration result."""
        inference_result = {
            "registration": "B-1234",
            "confidence": 0.95,
            "raw_text": "B-1234",
            "all_matches": [
                {"text": "B-1234", "confidence": 0.95}
            ],
            "yolo_boxes": [
                {
                    "class_id": 0,
                    "x_center": 0.5,
                    "y_center": 0.3,
                    "width": 0.2,
                    "height": 0.1,
                    "text": "B-1234",
                    "confidence": 0.95
                }
            ]
        }

        result = wrap_registration_result(inference_result)

        assert isinstance(result, RegistrationResult)
        assert result.registration == "B-1234"
        assert result.confidence == 0.95
        assert result.raw_text == "B-1234"
        assert len(result.yolo_boxes) == 1
        assert result.yolo_boxes[0].text == "B-1234"
        assert len(result.all_matches) == 1

    def test_wrap_minimal_registration_result(self):
        """Test wrapping a minimal registration result."""
        inference_result = {
            "registration": "B-5678",
            "confidence": 0.80,
            "raw_text": "B-5678"
        }

        result = wrap_registration_result(inference_result)

        assert result.registration == "B-5678"
        assert result.confidence == 0.80
        assert len(result.yolo_boxes) == 0
        assert len(result.all_matches) == 0
