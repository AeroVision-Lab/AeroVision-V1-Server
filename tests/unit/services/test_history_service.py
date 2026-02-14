"""
Unit tests for HistoryService.
"""

import os
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import sys

# Mock aerovision_inference before importing
mock_aerovision_inference = MagicMock()
mock_vector_record_class = MagicMock()
mock_aerovision_inference.VectorRecord = mock_vector_record_class
mock_aerovision_inference.ModelPredictor = MagicMock()
sys.modules['aerovision_inference'] = mock_aerovision_inference

# Set test environment before importing
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("MODEL_DIR", "tests/fixtures/models")
os.environ.setdefault("DEVICE", "cpu")


@pytest.fixture
def sample_historical_records(test_image_bytes):
    """Sample historical records for testing."""
    import base64
    image_base64 = base64.b64encode(test_image_bytes).decode('utf-8')

    return [
        {
            "id": "record_001",
            "image_url": "https://example.com/image1.jpg",
            "aircraft_type": "A320",
            "airline": "CEA",
            "aircraft_confidence": 0.85,
            "airline_confidence": 0.75,
            "timestamp": "2025-01-28T10:00:00Z",
            "metadata": {
                "image_data": f"data:image/jpeg;base64,{image_base64}"
            }
        },
        {
            "id": "record_002",
            "aircraft_type": "B738",
            "airline": "CSN",
            "aircraft_confidence": 0.90,
            "airline_confidence": 0.80,
            "metadata": {
                "image_data": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]


@pytest.fixture
def test_image_bytes():
    """Generate a test image as bytes."""
    image = Image.new("RGB", (640, 640), color=(100, 150, 200))
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr.read()


@pytest.fixture
def mock_vector_db():
    """Mock vector database."""
    mock_db = MagicMock()
    mock_db.add_record.return_value = True
    mock_db.search_similar.return_value = []
    return mock_db


@pytest.fixture
def mock_predictor():
    """Mock model predictor."""
    predictor = MagicMock()

    # Mock embedding methods
    mock_aircraft_model = MagicMock()
    mock_aircraft_embed = MagicMock(return_value=(np.random.rand(512),))
    mock_aircraft_model.embed = mock_aircraft_embed

    mock_airline_model = MagicMock()
    mock_airline_embed = MagicMock(return_value=(np.random.rand(512),))
    mock_airline_model.embed = mock_airline_embed

    predictor.aircraft_model = mock_aircraft_model
    predictor.airline_model = mock_airline_model
    predictor.device = "cpu"
    predictor.image_size = 640

    return predictor


@pytest.fixture
def mock_enhanced_predictor(mock_vector_db, mock_predictor):
    """Mock enhanced predictor with vector database."""
    enhanced = MagicMock()
    enhanced.vector_db = mock_vector_db
    enhanced.predictor = mock_predictor
    enhanced.get_db_statistics.return_value = {
        "available": True,
        "total_records": 100,
        "aircraft_types": {"A320": 50, "B738": 50},
        "airlines": {"CEA": 60, "CSN": 40}
    }
    return enhanced


class TestHistoryService:
    """Test cases for HistoryService."""

    def test_init_history_service(self):
        """Test HistoryService initialization."""
        from app.services.history_service import HistoryService

        service = HistoryService()
        assert service is not None
        assert hasattr(service, '_enhanced_predictor')

    @patch('app.services.history_service.InferenceFactory')
    def test_lazy_load_model_predictor(self, mock_factory, mock_enhanced_predictor):
        """Test lazy loading of model predictor."""
        # Mock the aerovision_inference module imports within the service
        mock_aerovision_inference.VectorRecord = MagicMock()
        mock_aerovision_inference.ModelPredictor = MagicMock(return_value=mock_enhanced_predictor)

        from app.services.history_service import HistoryService

        # Setup mock factory
        mock_factory.get_model_predictor.return_value = mock_enhanced_predictor

        service = HistoryService()
        result = service._get_model_predictor()

        assert result is not None
        # Note: The service doesn't use InferenceFactory, it uses aerovision_inference directly
        # So we check the predictor was created correctly instead
        mock_aerovision_inference.ModelPredictor.assert_called()

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    @patch('app.services.base.BaseService.load_image')
    def test_push_single_record_success(self, mock_load_image, mock_get_predictor, mock_enhanced_predictor, sample_historical_records, test_image_bytes):
        """Test pushing a single historical record successfully."""
        from PIL import Image
        import base64
        from app.services.history_service import HistoryService

        # Mock image loading to return a test image
        test_image = Image.open(BytesIO(test_image_bytes)).convert("RGB")
        mock_load_image.return_value = test_image

        # Ensure the mock predictor has all required attributes
        mock_get_predictor.return_value = mock_enhanced_predictor

        # Update sample data with valid base64
        sample_historical_records[0]['metadata']['image_data'] = f"data:image/jpeg;base64,{base64.b64encode(test_image_bytes).decode('utf-8')}"

        service = HistoryService()
        result = service.push_records(sample_historical_records[:1])

        assert result["success"] is True
        assert result["total"] == 1
        assert result["added"] == 1
        assert result["failed"] == 0

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    @patch('app.services.base.BaseService.load_image')
    def test_push_multiple_records_success(self, mock_load_image, mock_get_predictor, mock_enhanced_predictor, sample_historical_records, test_image_bytes):
        """Test pushing multiple historical records successfully."""
        from PIL import Image
        from app.services.history_service import HistoryService

        # Mock image loading to return a test image
        test_image = Image.open(BytesIO(test_image_bytes)).convert("RGB")
        mock_load_image.return_value = test_image

        mock_get_predictor.return_value = mock_enhanced_predictor

        service = HistoryService()
        result = service.push_records(sample_historical_records)

        assert result["success"] is True
        assert result["total"] == 2
        assert result["added"] == 2
        assert result["failed"] == 0

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    def test_push_record_without_image_data(self, mock_get_predictor, mock_enhanced_predictor):
        """Test pushing a record without image data (should fail)."""
        from app.services.history_service import HistoryService

        mock_get_predictor.return_value = mock_enhanced_predictor

        service = HistoryService()
        records = [
            {
                "id": "record_003",
                "aircraft_type": "A321",
                "airline": "CAC",
                "aircraft_confidence": 0.85,
                "airline_confidence": 0.75
            }
        ]
        result = service.push_records(records)

        assert result["success"] is True
        assert result["total"] == 1
        assert result["added"] == 0
        assert result["failed"] == 1

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    def test_get_statistics_success(self, mock_get_predictor, mock_enhanced_predictor):
        """Test getting statistics successfully."""
        from app.services.history_service import HistoryService

        mock_get_predictor.return_value = mock_enhanced_predictor

        service = HistoryService()
        stats = service.get_statistics()

        assert stats["available"] is True
        assert stats["total_records"] == 100
        assert stats["aircraft_types"] == {"A320": 50, "B738": 50}
        assert stats["airlines"] == {"CEA": 60, "CSN": 40}

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    def test_get_statistics_when_disabled(self, mock_get_predictor):
        """Test getting statistics when vector DB is disabled."""
        from app.services.history_service import HistoryService

        mock_get_predictor.return_value = None

        service = HistoryService()
        stats = service.get_statistics()

        assert stats["available"] is False
        assert stats["message"] == "向量数据库功能未启用"
