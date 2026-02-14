"""
Registration number OCR service.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.core import get_logger
from app.inference import InferenceFactory, wrap_registration_result
from app.schemas.registration import RegistrationResult
from app.services.base import BaseService

logger = get_logger("services.registration")


class RegistrationService(BaseService):
    """Service for registration number OCR."""

    def __init__(self):
        """Initialize the registration service."""
        self._ocr = None

    def _get_ocr(self):
        """Lazy load the registration OCR."""
        if self._ocr is None:
            self._ocr = InferenceFactory.get_registration_ocr()
        return self._ocr

    def recognize(self, image_input: str) -> tuple[RegistrationResult, float]:
        """
        Recognize registration number.

        Args:
            image_input: Base64 encoded image or URL

        Returns:
            Tuple of (registration result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        image = self.load_image(image_input)
        return self._recognize_image(image)

    def _recognize_image(self, image: Image.Image) -> tuple[RegistrationResult, float]:
        """
        Recognize registration number of a pre-loaded image.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (registration result, processing time ms)
        """
        ocr = self._get_ocr()

        def do_recognize():
            result = ocr.recognize(image)
            return wrap_registration_result(result)

        result, timing = self.measure_time(do_recognize)
        return result, timing

    async def recognize_batch(self, image_inputs: list[str]) -> list[dict[str, Any]]:
        """
        Recognize registration numbers from multiple images.

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
        registration_results = await loop.run_in_executor(None, self._recognize_batch, images)

        results = []
        for idx, result in enumerate(registration_results):
            if result is None:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": "Registration OCR failed"
                })
            else:
                results.append({
                    "index": idx,
                    "success": True,
                    "data": result.model_dump(by_alias=True),
                    "error": None
                })

        return results

    def _recognize_batch(self, images: list[Image.Image | None]) -> list[RegistrationResult]:
        """
        Recognize registration numbers from multiple pre-loaded images concurrently.

        Args:
            images: List of PIL Image objects (can contain None for failed loads)

        Returns:
            List of RegistrationResult objects
        """
        results = [None] * len(images)

        with ThreadPoolExecutor() as executor:
            def recognize_with_index(idx, image):
                if image is None:
                    return idx, None
                try:
                    result, _ = self._recognize_image(image)
                    return idx, result
                except Exception as e:
                    logger.error(f"Failed to recognize image at index {idx}: {e}")
                    return idx, None

            futures = [executor.submit(recognize_with_index, idx, img) for idx, img in enumerate(images)]

            for future in futures:
                idx, result = future.result()
                results[idx] = result

        return results
