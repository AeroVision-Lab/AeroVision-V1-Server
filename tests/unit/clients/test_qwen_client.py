"""
Unit tests for Qwen client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import numpy as np

from app.clients.qwen_client import QwenClient, QwenClientError


class TestQwenClient:
    """Test suite for QwenClient."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with test services."""
        from app.core.config import QwenServiceConfig

        settings = MagicMock()
        settings.qwen_services = [
            QwenServiceConfig(
                name="primary",
                url="http://localhost:8001/v1/chat/completions",
                model="qwen3.5-122b",
                timeout=30,
                enabled=True,
                priority=0,
            ),
            QwenServiceConfig(
                name="fallback",
                url="http://fallback:8001/v1/chat/completions",
                model="qwen3.5-plus",
                api_key="test-key",
                timeout=30,
                enabled=True,
                priority=1,
            ),
        ]
        return settings

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @pytest.fixture
    def test_prompt(self):
        """Test prompt."""
        return "Analyze this image and return JSON."

    @patch("app.clients.qwen_client.get_settings")
    def test_client_initialization(self, mock_get_settings, mock_settings):
        """Test that client initializes with services."""
        mock_get_settings.return_value = mock_settings

        client = QwenClient()
        assert client._services == mock_settings.qwen_services

    @patch("app.clients.qwen_client.get_settings")
    def test_client_no_services_raises(self, mock_get_settings):
        """Test that client raises if no services configured."""
        settings = MagicMock()
        settings.qwen_services = []
        mock_get_settings.return_value = settings

        with pytest.raises(QwenClientError, match="No enabled Qwen services"):
            QwenClient()

    @patch("app.clients.qwen_client.get_settings")
    @patch("app.clients.qwen_client.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_generate_success_primary(
        self, mock_client_class, mock_get_settings, mock_settings, test_image, test_prompt
    ):
        """Test successful generation from primary service."""
        mock_get_settings.return_value = mock_settings

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"violations": [], "suggestions": []}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        client = QwenClient()
        result, service_name = await client.generate(test_image, test_prompt)

        assert service_name == "primary"
        assert "violations" in result
        assert "suggestions" in result

    @patch("app.clients.qwen_client.get_settings")
    @patch("app.clients.qwen_client.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_generate_failover(
        self, mock_client_class, mock_get_settings, mock_settings, test_image, test_prompt
    ):
        """Test failover to secondary service when primary fails."""
        mock_get_settings.return_value = mock_settings

        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status.side_effect = Exception("Connection error")

        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"violations": [], "suggestions": []}'
                    }
                }
            ]
        }
        mock_response_success.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_response_fail, mock_response_success])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        client = QwenClient()
        result, service_name = await client.generate(test_image, test_prompt)

        assert service_name == "fallback"
        assert "violations" in result

    @patch("app.clients.qwen_client.get_settings")
    @patch("app.clients.qwen_client.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_generate_all_fail(
        self, mock_client_class, mock_get_settings, mock_settings, test_image, test_prompt
    ):
        """Test that QwenClientError is raised when all services fail."""
        mock_get_settings.return_value = mock_settings

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Connection error")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        client = QwenClient()

        with pytest.raises(QwenClientError, match="All Qwen services failed"):
            await client.generate(test_image, test_prompt)

    def test_parse_json_plain(self):
        """Test parsing plain JSON response."""
        content = '{"violations": [], "suggestions": []}'
        result = QwenClient._parse_json_response(content)
        assert "violations" in result
        assert "suggestions" in result

    def test_parse_json_with_fence(self):
        """Test parsing JSON wrapped in code fence."""
        content = '```json\n{"violations": [], "suggestions": []}\n```'
        result = QwenClient._parse_json_response(content)
        assert "violations" in result
        assert "suggestions" in result

    def test_parse_json_invalid_raises(self):
        """Test that invalid JSON raises ValueError."""
        content = "This is not JSON"
        with pytest.raises(ValueError, match="Cannot parse Qwen response"):
            QwenClient._parse_json_response(content)
