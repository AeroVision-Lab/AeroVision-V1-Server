"""
Tests for Qwen VLM-based quality detector.
"""

import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.quality.qwen_detector import QwenDetector, QwenResult
from app.clients.qwen_client import QwenClientError


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return Image.new("RGB", (800, 600), color=(100, 150, 200))


@pytest.fixture
def mock_qwen_response():
    """Mock Qwen API response."""
    return {
        "violations": [
            {
                "rule_id": "Rules1.2.2",
                "rule_name": "玻璃反光/眩光",
                "severity": "major",
                "confidence": 0.85,
                "description": "画面存在明显的玻璃反光"
            },
            {
                "rule_id": "Rules1.6.3",
                "rule_name": "构图问题",
                "severity": "minor",
                "confidence": 0.65,
                "description": "飞机在画面中位置偏离"
            }
        ],
        "suggestions": [
            "建议避免在玻璃窗内拍摄",
            "建议调整拍摄角度"
        ]
    }


@pytest.mark.asyncio
async def test_detect_with_violations(sample_image, mock_qwen_response):
    """Test detection with violations found."""
    with patch("app.services.quality.qwen_detector.QwenClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.generate = AsyncMock(return_value=(mock_qwen_response, "primary"))

        detector = QwenDetector()
        result = await detector.detect(sample_image)

        assert isinstance(result, QwenResult)
        assert len(result.violations) == 2
        assert result.violations[0].rule_id == "Rules1.2.2"
        assert result.violations[0].severity == "major"
        assert result.violations[0].source == "qwen"
        assert len(result.suggestions) == 2
        assert result.service_name == "primary"


@pytest.mark.asyncio
async def test_detect_no_violations(sample_image):
    """Test detection with no violations."""
    clean_response = {"violations": [], "suggestions": []}

    with patch("app.services.quality.qwen_detector.QwenClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.generate = AsyncMock(return_value=(clean_response, "primary"))

        detector = QwenDetector()
        result = await detector.detect(sample_image)

        assert len(result.violations) == 0
        assert len(result.suggestions) == 0


@pytest.mark.asyncio
async def test_detect_client_error(sample_image):
    """Test detection when Qwen client fails."""
    with patch("app.services.quality.qwen_detector.QwenClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.generate = AsyncMock(side_effect=QwenClientError("All services failed"))

        detector = QwenDetector()
        with pytest.raises(QwenClientError):
            await detector.detect(sample_image)


@pytest.mark.asyncio
async def test_parse_violations_invalid_data(sample_image):
    """Test parsing with invalid violation data."""
    invalid_response = {
        "violations": [
            {"rule_id": "Rules1.2.2", "rule_name": "test"},  # missing required fields
            {
                "rule_id": "Rules1.6.3",
                "rule_name": "构图问题",
                "severity": "minor",
                "confidence": 0.65,
                "description": "valid violation"
            }
        ],
        "suggestions": []
    }

    with patch("app.services.quality.qwen_detector.QwenClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.generate = AsyncMock(return_value=(invalid_response, "primary"))

        detector = QwenDetector()
        result = await detector.detect(sample_image)

        # Should skip invalid entry and only return valid one
        assert len(result.violations) == 1
        assert result.violations[0].rule_id == "Rules1.6.3"

