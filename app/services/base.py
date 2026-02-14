"""
Base service class for all services.

Provides common functionality for request processing, image loading,
and error handling.
"""

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.core.exceptions import ImageLoadError
from app.core.logging import logger


class BaseService:
    """Base class for all services with common utilities."""

    @staticmethod
    def load_image(image_input: str) -> Image.Image:
        """
        Load an image from base64 string or URL.

        Args:
            image_input: Base64 encoded image or URL

        Returns:
            PIL.Image: Loaded image

        Raises:
            ImageLoadError: If image loading fails
        """
        # Check if it's a URL
        if image_input.startswith(("http://", "https://")):
            return BaseService._load_from_url(image_input)

        # Check if it's base64 with data URL prefix
        if image_input.startswith("data:image/"):
            return BaseService._load_from_base64(image_input)

        # Try as raw base64 (without data URL prefix)
        try:
            return BaseService._load_from_base64(image_input)
        except Exception:
            raise ImageLoadError(f"Cannot load image from input: {image_input[:50]}...")

    @staticmethod
    def _load_from_url(url: str) -> Image.Image:
        """Load image from URL."""
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except httpx.HTTPError as e:
            raise ImageLoadError(f"Failed to load image from URL: {e}")

    @staticmethod
    def _load_from_base64(data: str) -> Image.Image:
        """Load image from base64 string."""
        try:
            # Remove data URL prefix if present
            if "," in data:
                data = data.split(",", 1)[1]

            image_bytes = base64.b64decode(data)
            return Image.open(BytesIO(image_bytes))
        except (ValueError, OSError) as e:
            raise ImageLoadError(f"Failed to decode base64 image: {e}")

    @staticmethod
    def measure_time(func, *args, **kwargs) -> tuple[Any, float]:
        """
        Measure execution time of a function.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, execution_time_ms)
        """
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result, (time.perf_counter() - start_time) * 1000
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs) -> Any:
        """
        Safely execute a function, returning default on error.

        Args:
            func: Function to execute
            *args: Function arguments
            default: Default value on error
            **kwargs: Function keyword arguments

        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Safe execute failed for {func.__name__}: {e}")
            return default
