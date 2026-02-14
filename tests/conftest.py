"""
Pytest configuration and fixtures for Aerovision-V1-Server tests.
"""

import os
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
import numpy as np
import io


# Set test environment before importing app
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("MODEL_DIR", "tests/fixtures/models")
os.environ.setdefault("DEVICE", "cpu")


@pytest.fixture
def test_image_path() -> Path:
    """Path to a test image."""
    return Path(__file__).parent / "fixtures" / "images" / "test_aircraft.jpg"


@pytest.fixture
def test_image_bytes() -> bytes:
    """Generate a test image as bytes."""
    image = Image.new("RGB", (640, 640), color=(100, 150, 200))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr.read()


@pytest.fixture
def test_image_base64(test_image_bytes: bytes) -> str:
    """Generate a test image as base64 string."""
    import base64
    return base64.b64encode(test_image_bytes).decode("utf-8")


@pytest.fixture
def sample_quality_result() -> dict:
    """Sample quality assessment result."""
    return {
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


@pytest.fixture
def sample_aircraft_result() -> dict:
    """Sample aircraft classification result."""
    return {
        "predictions": [
            {"class": "A320", "confidence": 0.85},
            {"class": "B738", "confidence": 0.10},
            {"class": "A321", "confidence": 0.05}
        ],
        "top1": {"class": "A320", "confidence": 0.85},
        "top_k": 5
    }


@pytest.fixture
def sample_airline_result() -> dict:
    """Sample airline classification result."""
    return {
        "predictions": [
            {"class": "CEA", "confidence": 0.75},
            {"class": "CSN", "confidence": 0.15},
            {"class": "CAC", "confidence": 0.10}
        ],
        "top1": {"class": "CEA", "confidence": 0.75},
        "top_k": 5
    }


@pytest.fixture
def sample_registration_result() -> dict:
    """Sample registration OCR result."""
    return {
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


@pytest.fixture
def mock_inference_factory():
    """Mock the InferenceFactory."""
    with patch("app.inference.factory.InferenceFactory") as mock:
        yield mock


@pytest.fixture
def mock_aircraft_classifier(sample_aircraft_result: dict):
    """Mock AircraftClassifier."""
    mock = MagicMock()
    mock.predict.return_value = sample_aircraft_result
    return mock


@pytest.fixture
def mock_airline_classifier(sample_airline_result: dict):
    """Mock AirlineClassifier."""
    mock = MagicMock()
    mock.predict.return_value = sample_airline_result
    return mock


@pytest.fixture
def mock_quality_assessor(sample_quality_result: dict):
    """Mock QualityAssessor."""
    mock = MagicMock()
    mock.assess.return_value = sample_quality_result
    mock.quick_assess.return_value = {
        "success": True,
        "pass": True,
        "score": sample_quality_result["details"]["sharpness"]
    }
    return mock


@pytest.fixture
def mock_registration_ocr(sample_registration_result: dict):
    """Mock RegistrationOCR."""
    mock = MagicMock()
    mock.recognize.return_value = sample_registration_result
    return mock


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing."""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        yield client
