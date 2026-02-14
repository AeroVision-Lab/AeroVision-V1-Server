"""
Quality assessment service.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.core import get_logger
from app.inference import InferenceFactory, wrap_quality_result
from app.schemas.quality import QualityResult
from app.services.base import BaseService

logger = get_logger("services.quality")


class QualityService(BaseService):
    """Service for image quality assessment."""

    def __init__(self):
        """Initialize the quality service."""
        self._assessor = None

    def _get_assessor(self):
        """Lazy load the quality assessor."""
        if self._assessor is None:
            self._assessor = InferenceFactory.get_quality_assessor()
        return self._assessor

    def assess(self, image_input: str) -> tuple[QualityResult, float]:
        """
        Assess image quality.

        Args:
            image_input: Base64 encoded image or URL

        Returns:
            Tuple of (quality result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        image = self.load_image(image_input)
        return self._assess_image(image)

    def _assess_image(self, image: Image.Image) -> tuple[QualityResult, float]:
        """
        Assess quality of a pre-loaded image.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (quality result, processing time ms)
        """
        assessor = self._get_assessor()

        def do_assess():
            result = assessor.assess(image)
            return wrap_quality_result(result)

        result, timing = self.measure_time(do_assess)
        return result, timing

    async def assess_batch(self, image_inputs: list[str]) -> list[dict[str, Any]]:
        """
        Assess quality of multiple images.

        Args:
            image_inputs: List of base64 encoded images or URLs

        Returns:
            List of results with index, success status, and data/error
        """
        import asyncio

        async def load_image_async(image_input: str):
            try:
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, lambda: self.load_image(image_input))
                return image
            except ImageLoadError:
                return None

        images = await asyncio.gather(*[load_image_async(img) for img in image_inputs])

        loop = asyncio.get_event_loop()
        quality_results = await loop.run_in_executor(None, self._assess_batch, images)

        results = []
        for idx, result in enumerate(quality_results):
            if result is None:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": "Quality assessment failed"
                })
            else:
                results.append({
                    "index": idx,
                    "success": True,
                    "data": result.model_dump(by_alias=True),
                    "error": None
                })

        return results

    def _assess_batch(self, images: list[Image.Image | None]) -> list[QualityResult]:
        """
        Assess quality of multiple pre-loaded images concurrently.

        Args:
            images: List of PIL Image objects (can contain None for failed loads)

        Returns:
            List of QualityResult objects
        """
        results = [None] * len(images)

        for i, image in enumerate(images):
            if image is None:
                results[i] = None
                continue
            try:
                result, _ = self._assess_image(image)
                results[i] = result
            except Exception as e:
                logger.error(f"Failed to assess image at index {i}: {e}")
                results[i] = None

        return results
