"""
Unit tests for AerovisionException handlers.
"""

import json

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AerovisionException,
    ImageLoadError,
    InferenceError,
    ModelNotLoadedError,
    ValidationError,
    RateLimitError,
)
from app.main import aerovision_exception_handler, global_exception_handler


class TestAerovisionExceptionHandler:
    """Tests for AerovisionException handler."""

    @pytest.mark.asyncio
    async def test_image_load_error_handler(self):
        """Test ImageLoadError returns 400 status code."""
        exc = ImageLoadError("Failed to load image")
        request = Request(scope={"type": "http"})

        response = await aerovision_exception_handler(request, exc)

        assert response.status_code == 400
        body = json.loads(response.body.decode())
        assert body["success"] is False
        assert body["error"]["code"] == "IMAGE_LOAD_ERROR"
        assert body["error"]["message"] == "Failed to load image"

    @pytest.mark.asyncio
    async def test_validation_error_handler(self):
        """Test ValidationError returns 422 status code."""
        exc = ValidationError("Invalid input")
        request = Request(scope={"type": "http"})

        response = await aerovision_exception_handler(request, exc)

        assert response.status_code == 422
        body = json.loads(response.body.decode())
        assert body["success"] is False
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Invalid input"

    @pytest.mark.asyncio
    async def test_model_not_loaded_error_handler(self):
        """Test ModelNotLoadedError returns 503 status code."""
        exc = ModelNotLoadedError("AircraftClassifier")
        request = Request(scope={"type": "http"})

        response = await aerovision_exception_handler(request, exc)

        assert response.status_code == 503
        body = json.loads(response.body.decode())
        assert body["success"] is False
        assert body["error"]["code"] == "MODEL_NOT_LOADED"
        assert "AircraftClassifier" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_inference_error_handler(self):
        """Test InferenceError returns 500 status code."""
        exc = InferenceError("Inference failed")
        request = Request(scope={"type": "http"})

        response = await aerovision_exception_handler(request, exc)

        assert response.status_code == 500
        body = json.loads(response.body.decode())
        assert body["success"] is False
        assert body["error"]["code"] == "INFERENCE_ERROR"
        assert body["error"]["message"] == "Inference failed"

    @pytest.mark.asyncio
    async def test_rate_limit_error_handler(self):
        """Test RateLimitError returns 429 status code."""
        exc = RateLimitError("Too many requests")
        request = Request(scope={"type": "http"})

        response = await aerovision_exception_handler(request, exc)

        assert response.status_code == 429
        body = json.loads(response.body.decode())
        assert body["success"] is False
        assert body["error"]["code"] == "RATE_LIMIT_ERROR"
        assert body["error"]["message"] == "Too many requests"

    @pytest.mark.asyncio
    async def test_custom_aerovision_exception_handler(self):
        """Test custom AerovisionException returns 500 status code (default)."""
        exc = AerovisionException("Custom error", code="CUSTOM_ERROR")
        request = Request(scope={"type": "http"})

        response = await aerovision_exception_handler(request, exc)

        assert response.status_code == 500
        body = json.loads(response.body.decode())
        assert body["success"] is False
        assert body["error"]["code"] == "CUSTOM_ERROR"
        assert body["error"]["message"] == "Custom error"


class TestGlobalExceptionHandler:
    """Tests for global exception handler."""

    @pytest.mark.asyncio
    async def test_generic_exception_handler(self):
        """Test generic exception returns 500 status code."""
        exc = Exception("Unexpected error")
        request = Request(scope={"type": "http"})

        response = await global_exception_handler(request, exc)

        assert response.status_code == 500
        body = json.loads(response.body.decode())
        assert body["success"] is False
        assert body["error"] == "内部服务错误"
