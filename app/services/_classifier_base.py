"""
Base class for classification services.

Provides common functionality for aircraft and airline classification services.
"""

import asyncio
from typing import Any, Callable, TypeVar

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.core.logging import get_logger
from app.services.base import BaseService

TResult = TypeVar('TResult')


class ClassificationServiceBase(BaseService):
    """Base class for classification services with common batch processing logic."""

    def __init__(
        self,
        classifier_factory: Callable,
        result_wrapper: Callable,
        result_type: type[TResult],
        service_name: str,
        error_message: str
    ):
        """
        Initialize the classification service.

        Args:
            classifier_factory: Factory function to create/get the classifier
            result_wrapper: Function to wrap raw inference results
            result_type: Type of the result class
            service_name: Name for logging purposes
            error_message: Error message for failed classifications
        """
        self._classifier = None
        self._classifier_factory = classifier_factory
        self._result_wrapper = result_wrapper
        self._result_type = result_type
        self._service_name = service_name
        self._error_message = error_message

    def _get_classifier(self):
        """Lazy load the classifier."""
        if self._classifier is None:
            self._classifier = self._classifier_factory()
        return self._classifier

    def classify(self, image_input: str, top_k: int | None = None) -> tuple[TResult, float]:
        """
        Classify image.

        Args:
            image_input: Base64 encoded image or URL
            top_k: Number of top predictions to return

        Returns:
            Tuple of (result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        image = self.load_image(image_input)
        return self._classify_image(image, top_k)

    def _classify_image(self, image: Image.Image, top_k: int | None = None) -> tuple[TResult, float]:
        """
        Classify a pre-loaded image.

        Args:
            image: PIL Image object
            top_k: Number of top predictions to return

        Returns:
            Tuple of (result, processing time ms)
        """
        classifier = self._get_classifier()

        def do_classify():
            result = classifier.predict(image, top_k=top_k)
            return self._result_wrapper(result)

        result, timing = self.measure_time(do_classify)
        return result, timing

    async def classify_batch(
        self,
        image_inputs: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Classify multiple images.

        Args:
            image_inputs: List of base64 encoded images or URLs
            top_k: Number of top predictions to return

        Returns:
            List of results with index, success status, and data/error
        """
        async def load_image_async(image_input: str):
            try:
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, lambda: self.load_image(image_input))
                return image
            except ImageLoadError:
                return None

        images = await asyncio.gather(*[load_image_async(img) for img in image_inputs])

        results = await self._classify_batch(images, top_k)

        output = []
        for idx, result in enumerate(results):
            if result is None:
                output.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": self._error_message
                })
            else:
                output.append({
                    "index": idx,
                    "success": True,
                    "data": result.model_dump(by_alias=True),
                    "error": None
                })

        return output

    async def _classify_batch(self, images: list[Image.Image | None], top_k: int | None = None) -> list[TResult]:
        """
        Classify multiple pre-loaded images using batch inference.

        Args:
            images: List of PIL Image objects (can contain None for failed loads)
            top_k: Number of top predictions to return

        Returns:
            List of result objects
        """
        valid_images = [(idx, img) for idx, img in enumerate(images) if img is not None]

        if not valid_images:
            return [None] * len(images)

        indices, batch_images = zip(*sorted(valid_images, key=lambda x: x[0]))

        def run_batch_prediction():
            classifier = self._get_classifier()
            batch_results = classifier.predict(list(batch_images), top_k=top_k)
            return batch_results

        loop = asyncio.get_event_loop()
        batch_results = await loop.run_in_executor(None, run_batch_prediction)

        results = [None] * len(images)
        logger = get_logger(self._service_name)

        for idx, result in zip(indices, batch_results):
            if result is not None:
                try:
                    wrapped_result = self._result_wrapper(result)
                    results[idx] = wrapped_result
                except Exception as e:
                    logger.error(f"Failed to wrap result at index {idx}: {e}")
                    results[idx] = None

        return results
