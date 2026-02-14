"""
Integration tests for history API endpoints.
"""

import os
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from httpx import ASGITransport, AsyncClient
import numpy as np
import sys

# Mock aerovision_inference before importing
mock_aerovision_inference = MagicMock()
sys.modules['aerovision_inference'] = mock_aerovision_inference

# Set test environment
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("MODEL_DIR", "tests/fixtures/models")
os.environ.setdefault("DEVICE", "cpu")


@pytest.fixture
async def async_client():
    """Async HTTP client for testing."""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def sample_historical_records():
    """Sample historical records for API testing."""
    return [
        {
            "id": "record_api_001",
            "image_url": "https://example.com/image1.jpg",
            "aircraft_type": "A320",
            "airline": "CEA",
            "aircraft_confidence": 0.85,
            "airline_confidence": 0.75,
            "timestamp": "2025-01-28T10:00:00Z",
            "metadata": {
                "image_data": "data:image/jpeg;base64,/9j/4AAQ..."
            }
        },
        {
            "id": "record_api_002",
            "aircraft_type": "B738",
            "airline": "CSN",
            "aircraft_confidence": 0.90,
            "airline_confidence": 0.80,
            "metadata": {
                "image_data": "data:image/jpeg;base64,/9j/4AAQ..."
            }
        }
    ]


class TestHistoryAPI:
    """Integration tests for history API endpoints."""

    @patch('app.services.base.BaseService.load_image')
    @patch('app.services.history_service.HistoryService._get_model_predictor')
    async def test_push_historical_records_success(self, mock_get_predictor, mock_load_image, async_client, sample_historical_records):
        """Test pushing historical records successfully."""
        from PIL import Image
        from io import BytesIO
        import base64

        # Mock image loading
        image = Image.new("RGB", (640, 640), color=(100, 150, 200))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        test_image_bytes = img_byte_arr.read()
        test_image = Image.open(BytesIO(test_image_bytes)).convert("RGB")
        mock_load_image.return_value = test_image

        # Mock the enhanced predictor
        mock_enhanced_predictor = MagicMock()
        mock_vector_db = MagicMock()
        mock_vector_db.add_record.return_value = True
        mock_enhanced_predictor.vector_db = mock_vector_db

        mock_predictor = MagicMock()
        mock_aircraft_model = MagicMock()
        mock_aircraft_model.embed = MagicMock(return_value=(np.random.rand(512),))
        mock_airline_model = MagicMock()
        mock_airline_model.embed = MagicMock(return_value=(np.random.rand(512),))
        mock_predictor.aircraft_model = mock_aircraft_model
        mock_predictor.airline_model = mock_airline_model
        mock_predictor.device = "cpu"
        mock_predictor.image_size = 640
        mock_enhanced_predictor.predictor = mock_predictor

        mock_get_predictor.return_value = mock_enhanced_predictor

        # Make request
        response = await async_client.post(
            "/api/v1/history/push",
            json=sample_historical_records
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == len(sample_historical_records)
        assert data["added"] == len(sample_historical_records)
        assert data["failed"] == 0

    @patch('app.services.base.BaseService.load_image')
    @patch('app.services.history_service.HistoryService._get_model_predictor')
    async def test_push_historical_records_partial_failure(self, mock_get_predictor, mock_load_image, async_client):
        """Test pushing records with some failures."""
        from PIL import Image
        from io import BytesIO
        import base64

        # Mock image loading
        image = Image.new("RGB", (640, 640), color=(100, 150, 200))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        test_image_bytes = img_byte_arr.read()
        test_image = Image.open(BytesIO(test_image_bytes)).convert("RGB")
        image_base64 = base64.b64encode(test_image_bytes).decode('utf-8')
        mock_load_image.return_value = test_image

        # Mock the enhanced predictor
        mock_enhanced_predictor = MagicMock()
        mock_vector_db = MagicMock()
        mock_vector_db.add_record.return_value = True
        mock_enhanced_predictor.vector_db = mock_vector_db

        mock_predictor = MagicMock()
        mock_aircraft_model = MagicMock()
        mock_aircraft_model.embed = MagicMock(return_value=(np.random.rand(512),))
        mock_airline_model = MagicMock()
        mock_airline_model.embed = MagicMock(return_value=(np.random.rand(512),))
        mock_predictor.aircraft_model = mock_aircraft_model
        mock_predictor.airline_model = mock_airline_model
        mock_predictor.device = "cpu"
        mock_predictor.image_size = 640
        mock_enhanced_predictor.predictor = mock_predictor

        mock_get_predictor.return_value = mock_enhanced_predictor

        # Create records with one missing image data
        records = [
            {
                "id": "record_001",
                "image_url": "https://example.com/image1.jpg",
                "aircraft_type": "A320",
                "airline": "CEA",
                "aircraft_confidence": 0.85,
                "airline_confidence": 0.75,
                "metadata": {
                    "image_data": f"data:image/jpeg;base64,{image_base64}"
                }
            },
            {
                "id": "record_002",
                "aircraft_type": "B738",
                "airline": "CSN",
                "aircraft_confidence": 0.90,
                "airline_confidence": 0.80
                # Missing image_data
            }
        ]

        response = await async_client.post(
            "/api/v1/history/push",
            json=records
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 2
        assert data["added"] == 1
        assert data["failed"] == 1

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    async def test_push_historical_records_without_enhanced_predictor(self, mock_get_predictor, async_client):
        """Test pushing records when enhanced predictor is not available."""
        # Mock predictor to return None
        mock_get_predictor.return_value = None

        records = [
            {
                "id": "record_001",
                "image_url": "https://example.com/image1.jpg",
                "aircraft_type": "A320",
                "airline": "CEA",
                "aircraft_confidence": 0.85,
                "airline_confidence": 0.75
            }
        ]

        response = await async_client.post(
            "/api/v1/history/push",
            json=records
        )

        assert response.status_code == 503
        data = response.json()
        assert "向量数据库功能未启用" in str(data.get("detail", ""))

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    async def test_get_history_stats_success(self, mock_get_predictor, async_client):
        """Test getting history statistics successfully."""
        # Mock the enhanced predictor
        mock_enhanced_predictor = MagicMock()
        mock_vector_db = MagicMock()
        mock_enhanced_predictor.get_db_statistics.return_value = {
            "available": True,
            "total_records": 100,
            "aircraft_types": {"A320": 50, "B738": 50},
            "airlines": {"CEA": 60, "CSN": 40}
        }
        mock_enhanced_predictor.vector_db = mock_vector_db
        mock_enhanced_predictor.predictor = MagicMock()

        mock_get_predictor.return_value = mock_enhanced_predictor

        response = await async_client.get("/api/v1/history/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["total_records"] == 100
        assert data["aircraft_types"]["A320"] == 50
        assert data["airlines"]["CEA"] == 60

    @patch('app.services.history_service.HistoryService._get_model_predictor')
    async def test_get_history_stats_when_disabled(self, mock_get_predictor, async_client):
        """Test getting statistics when vector DB is disabled."""
        # Mock predictor to return None
        mock_get_predictor.return_value = None

        response = await async_client.get("/api/v1/history/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert data["message"] == "向量数据库功能未启用"
