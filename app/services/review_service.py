"""
Aggregated review service.
"""

import asyncio
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from app.schemas.review import ReviewResult, ReviewQualityResult, ReviewAircraftResult, ReviewAirlineResult, ReviewRegistrationResult
from app.schemas.quality import QualityResult
from app.schemas.aircraft import AircraftResult
from app.schemas.airline import AirlineResult
from app.schemas.registration import RegistrationResult
from app.core.exceptions import ImageLoadError
from app.services.quality_service import QualityService
from app.services.aircraft_service import AircraftService
from app.services.airline_service import AirlineService
from app.services.registration_service import RegistrationService
from app.services.base import BaseService


class ReviewService(BaseService):
    """Service for aggregated image review."""

    def __init__(self):
        """Initialize the review service with sub-services."""
        self.quality_service = QualityService()
        self.aircraft_service = AircraftService()
        self.airline_service = AirlineService()
        self.registration_service = RegistrationService()

    def review(
        self,
        image_input: str,
        include_quality: bool = True,
        include_aircraft: bool = True,
        include_airline: bool = True,
        include_registration: bool = True
    ) -> tuple[ReviewResult, float]:
        """
        Perform a complete review of an image.

        Args:
            image_input: Base64 encoded image or URL
            include_quality: Whether to include quality assessment
            include_aircraft: Whether to include aircraft classification
            include_airline: Whether to include airline classification
            include_registration: Whether to include registration OCR

        Returns:
            Tuple of (review result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        start_time = self._now()

        # Load image once
        image = self.load_image(image_input)

        # Collect results
        quality_result = None
        aircraft_result = None
        airline_result = None
        registration_result = None

        if include_quality:
            quality_data_tuple = self.safe_execute(
                self.quality_service._assess_image, image
            )
            if quality_data_tuple:
                quality_data, _ = quality_data_tuple
                quality_result = ReviewQualityResult(
                    score=quality_data.score,
                    **{"pass": quality_data.pass_},
                    details=quality_data.details
                )

        if include_aircraft:
            aircraft_data_tuple = self.safe_execute(
                self.aircraft_service._classify_image, image
            )
            if aircraft_data_tuple:
                aircraft_data, _ = aircraft_data_tuple
                aircraft_result = ReviewAircraftResult(
                    type_code=aircraft_data.top1.class_,
                    confidence=aircraft_data.top1.confidence
                )

        if include_airline:
            airline_data_tuple = self.safe_execute(
                self.airline_service._classify_image, image
            )
            if airline_data_tuple:
                airline_data, _ = airline_data_tuple
                airline_result = ReviewAirlineResult(
                    airline_code=airline_data.top1.class_,
                    confidence=airline_data.top1.confidence
                )

        if include_registration:
            reg_data_tuple = self.safe_execute(
                self.registration_service._recognize_image, image
            )
            if reg_data_tuple:
                reg_data, _ = reg_data_tuple
                # Use registration confidence as clarity
                registration_result = ReviewRegistrationResult(
                    registration=reg_data.registration,
                    confidence=reg_data.confidence,
                    clarity=reg_data.confidence  # Using OCR confidence as proxy
                )

        # Build final result
        # Quality is mandatory - provide default if failed
        if quality_result is None:
            quality_result = ReviewQualityResult.model_validate({
                "score": 0.0,
                "pass": False,
                "details": None
            })

        # Aircraft is mandatory - provide default if failed
        if aircraft_result is None:
            aircraft_result = ReviewAircraftResult(
                type_code="UNKNOWN",
                confidence=0.0
            )

        result = ReviewResult(
            quality=quality_result,
            aircraft=aircraft_result,
            airline=airline_result,
            registration=registration_result
        )

        timing = (self._now() - start_time) * 1000
        return result, timing

    async def review_batch(
        self,
        image_inputs: list[str],
        include_quality: bool = True,
        include_aircraft: bool = True,
        include_airline: bool = True,
        include_registration: bool = True
    ) -> list[dict[str, Any]]:
        """
        Review multiple images using concurrent inference.

        Args:
            image_inputs: List of base64 encoded images or URLs
            include_quality: Whether to include quality assessment
            include_aircraft: Whether to include aircraft classification
            include_airline: Whether to include airline classification
            include_registration: Whether to include registration OCR

        Returns:
            List of results with index, success status, and data/error
        """
        # Load all images in parallel
        async def load_image_async(image_input: str):
            try:
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, lambda: self.load_image(image_input))
                return image, None
            except ImageLoadError as e:
                return None, str(e)
            except Exception as e:
                return None, str(e)

        loaded_results = await asyncio.gather(*[load_image_async(img) for img in image_inputs])
        images = [r[0] for r in loaded_results]
        image_errors = [r[1] for r in loaded_results]

        # Collect results from all services using async concurrent execution
        tasks = []

        # Create async tasks for all services
        if include_quality:
            # Quality service uses sync _assess_batch, wrap it in executor
            async def run_quality():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.quality_service._assess_batch, images)
            tasks.append(('quality', run_quality()))

        if include_aircraft:
            tasks.append(('aircraft', self.aircraft_service._classify_batch(images)))

        if include_airline:
            tasks.append(('airline', self.airline_service._classify_batch(images)))

        if include_registration:
            # Registration service uses sync _recognize_batch, wrap it in executor
            async def run_registration():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.registration_service._recognize_batch, images)
            tasks.append(('registration', run_registration()))

        # Execute tasks concurrently using asyncio.gather
        task_map = {name: coro for name, coro in tasks}
        results_list = await asyncio.gather(*task_map.values(), return_exceptions=True)

        results_map = {}
        for task_name, result in zip(task_map.keys(), results_list):
            if isinstance(result, Exception):
                from app.core.logging import get_logger
                logger = get_logger("review_service")
                logger.error(f"Failed to execute {task_name} task: {result}")
                results_map[task_name] = None
            else:
                results_map[task_name] = result

        # Extract individual results
        quality_results = results_map.get('quality')
        aircraft_results = results_map.get('aircraft')
        airline_results = results_map.get('airline')
        registration_results = results_map.get('registration')

        # Build final results
        results = []
        for idx, image_input in enumerate(image_inputs):
            # Check for image load errors
            if image_errors[idx] is not None:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": image_errors[idx]
                })
                continue

            try:
                # Get results for this index
                quality_result = quality_results[idx] if (quality_results is not None and idx < len(quality_results)) else None
                aircraft_result = aircraft_results[idx] if (aircraft_results is not None and idx < len(aircraft_results)) else None
                airline_result = airline_results[idx] if (airline_results is not None and idx < len(airline_results)) else None
                registration_result = registration_results[idx] if (registration_results is not None and idx < len(registration_results)) else None

                # Build result objects
                review_quality = None
                if include_quality:
                    if quality_result is not None:
                        review_quality = ReviewQualityResult(
                            score=quality_result.score,
                            **{"pass": quality_result.pass_},
                            details=quality_result.details
                        )
                    else:
                        review_quality = ReviewQualityResult.model_validate({
                            "score": 0.0,
                            "pass": False,
                            "details": None
                        })

                review_aircraft = None
                if include_aircraft:
                    if aircraft_result is not None:
                        review_aircraft = ReviewAircraftResult(
                            type_code=aircraft_result.top1.class_,
                            confidence=aircraft_result.top1.confidence
                        )
                    else:
                        review_aircraft = ReviewAircraftResult(
                            type_code="UNKNOWN",
                            confidence=0.0
                        )

                review_airline = None
                if include_airline:
                    if airline_result is not None:
                        review_airline = ReviewAirlineResult(
                            airline_code=airline_result.top1.class_,
                            confidence=airline_result.top1.confidence
                        )

                review_registration = None
                if include_registration:
                    if registration_result is not None:
                        review_registration = ReviewRegistrationResult(
                            registration=registration_result.registration,
                            confidence=registration_result.confidence,
                            clarity=registration_result.confidence
                        )

                result = ReviewResult(
                    quality=review_quality,
                    aircraft=review_aircraft,
                    airline=review_airline,
                    registration=review_registration
                )

                results.append({
                    "index": idx,
                    "success": True,
                    "data": result.model_dump(by_alias=True),
                    "error": None
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": f"Review failed: {e}"
                })

        return results

    @staticmethod
    def _now() -> float:
        """Get current time in seconds."""
        return time.perf_counter()
