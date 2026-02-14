"""
Unit tests for BaseService.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.exceptions import ImageLoadError
from app.services.base import BaseService


class TestBaseService:
    """Tests for BaseService."""

    def test_load_image_from_base64_with_prefix(self):
        """Test loading image from base64 with data URL prefix."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{img_b64}"

        result = BaseService.load_image(data_url)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_load_image_from_raw_base64(self):
        """Test loading image from raw base64 string."""
        img = Image.new("RGB", (50, 50), color="blue")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        result = BaseService.load_image(img_b64)

        assert isinstance(result, Image.Image)
        assert result.size == (50, 50)

    def test_load_image_invalid_base64_fails(self):
        """Test loading image from invalid base64 fails."""
        with pytest.raises(ImageLoadError):
            BaseService.load_image("not-a-valid-base64!!!")

    def test_measure_time_success(self):
        """Test measure_time returns result and timing."""
        def dummy_func(x):
            return x * 2

        result, timing = BaseService.measure_time(dummy_func, 5)

        assert result == 10
        assert timing >= 0
        assert timing < 100  # Should be very fast

    def test_measure_time_exception(self):
        """Test measure_time propagates exceptions."""
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            BaseService.measure_time(failing_func)

    def test_safe_execute_success(self):
        """Test safe_execute returns result on success."""
        result = BaseService.safe_execute(lambda x: x + 1, 5)
        assert result == 6

    def test_safe_execute_failure_returns_default(self):
        """Test safe_execute returns default on error."""
        result = BaseService.safe_execute(
            lambda: (_ for _ in ()).throw(ValueError("error")),
            default="default_value"
        )
        assert result == "default_value"

    @patch("app.services.base.httpx.get")
    def test_load_image_from_url(self, mock_get):
        """Test loading image from URL."""
        # Create mock response
        img = Image.new("RGB", (100, 100), color="green")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        mock_response = MagicMock()
        mock_response.content = buffer.getvalue()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = BaseService.load_image("https://example.com/test.jpg")

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
        mock_get.assert_called_once()

    @patch("app.services.base.httpx.get")
    def test_load_image_from_url_http_error(self, mock_get):
        """Test loading image from URL with HTTP error."""
        import httpx
        mock_get.side_effect = httpx.HTTPError("Network error")

        with pytest.raises(ImageLoadError, match="Failed to load image from URL"):
            BaseService.load_image("https://example.com/test.jpg")
