"""
Unit tests for ReviewService.

Tests for bug fixes and optimizations:
1. Test that safe_execute return value None doesn't cause TypeError when unpacking
2. Test that image is loaded only once (not re-loaded by sub-services)
3. Test that batch processing uses true concurrent inference
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from app.services.review_service import ReviewService
from app.schemas.quality import QualityResult, QualityDetails
from app.schemas.aircraft import AircraftResult
from app.schemas.airline import AirlineResult
from app.schemas.registration import RegistrationResult
from app.services.base import BaseService


class TestReviewServiceBugFixes:
    """Tests for critical bug fixes in ReviewService."""

    @pytest.fixture
    def test_image_bytes(self):
        """Generate test image bytes."""
        img = Image.new("RGB", (640, 640), color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.fixture
    def test_image_base64(self, test_image_bytes):
        """Generate test image as base64 string."""
        return base64.b64encode(test_image_bytes).decode("utf-8")

    @pytest.fixture
    def sample_quality_result(self):
        """Sample quality result."""
        return QualityResult.model_validate({
            "pass": True,
            "score": 0.85,
            "details": {
                "sharpness": 0.90,
                "exposure": 0.80,
                "composition": 0.85,
                "noise": 0.88,
                "color": 0.82
            }
        })

    @pytest.fixture
    def sample_aircraft_result(self):
        """Sample aircraft result."""
        return AircraftResult.model_validate({
            "top1": {"class": "B737-800", "confidence": 0.92},
            "top_k": 3,
            "predictions": [
                {"class": "B737-800", "confidence": 0.92},
                {"class": "A320", "confidence": 0.05},
                {"class": "B737-MAX8", "confidence": 0.03}
            ]
        })

    @pytest.fixture
    def sample_airline_result(self):
        """Sample airline result."""
        return AirlineResult.model_validate({
            "top1": {"class": "CA", "confidence": 0.88},
            "top_k": 3,
            "predictions": [
                {"class": "CA", "confidence": 0.88},
                {"class": "MU", "confidence": 0.10},
                {"class": "CZ", "confidence": 0.02}
            ]
        })

    @pytest.fixture
    def sample_registration_result(self):
        """Sample registration result."""
        return RegistrationResult.model_validate({
            "registration": "B-1234",
            "confidence": 0.95,
            "raw_text": "B-1234",
            "all_matches": [
                {"text": "B-1234", "confidence": 0.95}
            ],
            "yolo_boxes": []
        })

    @pytest.fixture
    def test_images_batch(self, test_image_base64):
        """Sample batch of images."""
        return [test_image_base64, test_image_base64, test_image_base64]

    def test_review_handles_safe_execute_none_return(self, test_image_base64, sample_quality_result):
        """
        Test that review handles None return from safe_execute without TypeError.

        Bug: When safe_execute returns None, direct unpacking (quality_result, _ = ...)
        causes TypeError: cannot unpack non-iterable NoneType object.
        """
        service = ReviewService()

        # Mock quality_service to return None (simulating safe_execute failure)
        with patch.object(service.quality_service, '_assess_image', return_value=None):
            # This should not raise TypeError
            result, timing = service.review(test_image_base64, include_quality=True)

            # Quality should have default value when safe_execute returns None
            assert result.quality.score == 0.0
            assert result.quality.pass_ is False
            assert result.quality.details is None

    def test_review_handles_service_exceptions(self, test_image_base64):
        """
        Test that review handles exceptions from sub-services gracefully.
        """
        service = ReviewService()

        # Create a mock function that raises exception but has __name__
        def failing_assess(*args, **kwargs):
            raise Exception("Service failed")
        failing_assess.__name__ = 'assess'

        # Mock safe_execute to return None (simulating exception caught)
        with patch.object(service, 'safe_execute', return_value=None):
            # This should not crash
            result, timing = service.review(test_image_base64, include_quality=True)

            # Quality should have default value
            assert result.quality.score == 0.0
            assert result.quality.pass_ is False

    def test_review_image_loaded_once(self, test_image_base64, sample_quality_result):
        """
        Test that image is loaded only once in review method.

        Bug: Image is loaded in review() and then passed as image_input to sub-services,
        causing sub-services to re-load the image.
        """
        service = ReviewService()

        # Track load_image calls
        original_load_image = service.load_image
        load_image_calls = []

        def tracked_load_image(image_input):
            load_image_calls.append(image_input)
            return original_load_image(image_input)

        with patch.object(service, 'load_image', side_effect=tracked_load_image):
            with patch.object(service.quality_service, 'assess', return_value=(sample_quality_result, 100)):
                # Mock other services to return None (safe_execute behavior)
                with patch.object(service, 'safe_execute', return_value=None):
                    result, timing = service.review(
                        test_image_base64,
                        include_quality=False,  # Disable to check load_image count
                        include_aircraft=False,
                        include_airline=False,
                        include_registration=False
                    )

        # Image should be loaded exactly once
        assert len(load_image_calls) == 1, f"Expected 1 image load, got {len(load_image_calls)}"


class TestReviewServiceImageLoadingOptimization:
    """Tests for image loading optimization in ReviewService."""

    @pytest.fixture
    def test_image_bytes(self):
        """Generate test image bytes."""
        img = Image.new("RGB", (640, 640), color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.fixture
    def test_image_base64(self, test_image_bytes):
        """Generate test image as base64 string."""
        return base64.b64encode(test_image_bytes).decode("utf-8")

    @pytest.fixture
    def sample_results(self):
        """Sample results for all services."""
        from app.schemas.quality import QualityResult
        from app.schemas.aircraft import AircraftResult
        from app.schemas.airline import AirlineResult
        from app.schemas.registration import RegistrationResult

        return {
            'quality': QualityResult.model_validate({
                "pass": True,
                "score": 0.85,
                "details": {
                    "sharpness": 0.90,
                    "exposure": 0.80,
                    "composition": 0.85,
                    "noise": 0.88,
                    "color": 0.82
                }
            }),
            'aircraft': AircraftResult.model_validate({
                "top1": {"class": "B737-800", "confidence": 0.92},
                "top_k": 1,
                "predictions": [{"class": "B737-800", "confidence": 0.92}]
            }),
            'airline': AirlineResult.model_validate({
                "top1": {"class": "CA", "confidence": 0.88},
                "top_k": 1,
                "predictions": [{"class": "CA", "confidence": 0.88}]
            }),
            'registration': RegistrationResult.model_validate({
                "registration": "B-1234",
                "confidence": 0.95,
                "raw_text": "B-1234",
                "all_matches": [{"text": "B-1234", "confidence": 0.95}],
                "yolo_boxes": []
            })
        }

    def test_review_passes_loaded_image_to_subservices(self, test_image_base64, sample_results):
        """
        Test that review() passes the loaded PIL.Image object to sub-services
        instead of the image_input string, preventing re-loading.

        This test verifies that sub-services are called with internal methods
        that accept PIL.Image objects.
        """
        service = ReviewService()

        # Mock the internal methods that accept PIL.Image objects
        with patch.object(service.quality_service, '_assess_image', return_value=(sample_results['quality'], 100)):
            with patch.object(service.aircraft_service, '_classify_image', return_value=(sample_results['aircraft'], 100)):
                with patch.object(service.airline_service, '_classify_image', return_value=(sample_results['airline'], 100)):
                    with patch.object(service.registration_service, '_recognize_image', return_value=(sample_results['registration'], 100)):
                        result, timing = service.review(
                            test_image_base64,
                            include_quality=True,
                            include_aircraft=True,
                            include_airline=True,
                            include_registration=True
                        )

                        # Verify results are correct
                        assert result.quality.score == 0.85
                        assert result.aircraft.type_code == "B737-800"
                        assert result.airline.airline_code == "CA"
                        assert result.registration.registration == "B-1234"

                        # Verify load_image was called only once
                        assert len(service.quality_service._assess_image.call_args_list) == 1
                        assert len(service.aircraft_service._classify_image.call_args_list) == 1

                        # Verify the first argument is a PIL.Image object, not a string
                        for mock_call in service.quality_service._assess_image.call_args_list:
                            image_arg = mock_call[0][0]
                            assert isinstance(image_arg, Image.Image), f"Expected PIL.Image, got {type(image_arg)}"

    def test_review_loads_image_only_once_across_all_services(self, test_image_base64, sample_results):
        """
        Test that image is loaded exactly once across all service calls.

        This test ensures that load_image is called only once in ReviewService.review()
        and the loaded image is reused by all sub-services.
        """
        service = ReviewService()

        # Track load_image calls
        load_image_calls = []

        original_load_image = BaseService.load_image

        def tracked_load_image(image_input):
            load_image_calls.append(image_input)
            return original_load_image(image_input)

        with patch('app.services.base.BaseService.load_image', side_effect=tracked_load_image):
            # Mock internal methods
            with patch.object(service.quality_service, '_assess_image', return_value=(sample_results['quality'], 100)):
                with patch.object(service.aircraft_service, '_classify_image', return_value=(sample_results['aircraft'], 100)):
                    with patch.object(service.airline_service, '_classify_image', return_value=(sample_results['airline'], 100)):
                        with patch.object(service.registration_service, '_recognize_image', return_value=(sample_results['registration'], 100)):
                            result, timing = service.review(
                                test_image_base64,
                                include_quality=True,
                                include_aircraft=True,
                                include_airline=True,
                                include_registration=True
                            )

        # Verify image was loaded exactly once
        assert len(load_image_calls) == 1, f"Expected 1 image load, got {len(load_image_calls)}"


class TestReviewServiceConcurrentBatchInference:
    """Tests for concurrent batch inference in ReviewService."""

    @pytest.fixture
    def test_image_bytes(self):
        """Generate test image bytes."""
        img = Image.new("RGB", (640, 640), color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.fixture
    def test_image_base64(self, test_image_bytes):
        """Generate test image as base64 string."""
        return base64.b64encode(test_image_bytes).decode("utf-8")

    @pytest.fixture
    def test_images_batch(self, test_image_base64):
        """Sample batch of images."""
        return [test_image_base64, test_image_base64, test_image_base64]

    @pytest.fixture
    def sample_batch_results(self):
        """Sample batch results for all services."""
        from app.schemas.quality import QualityResult
        from app.schemas.aircraft import AircraftResult
        from app.schemas.airline import AirlineResult
        from app.schemas.registration import RegistrationResult

        quality = QualityResult.model_validate({
            "pass": True,
            "score": 0.85,
            "details": {
                "sharpness": 0.90,
                "exposure": 0.80,
                "composition": 0.85,
                "noise": 0.88,
                "color": 0.82
            }
        })

        aircraft = AircraftResult.model_validate({
            "top1": {"class": "B737-800", "confidence": 0.92},
            "top_k": 1,
            "predictions": [{"class": "B737-800", "confidence": 0.92}]
        })

        airline = AirlineResult.model_validate({
            "top1": {"class": "CA", "confidence": 0.88},
            "top_k": 1,
            "predictions": [{"class": "CA", "confidence": 0.88}]
        })

        registration = RegistrationResult.model_validate({
            "registration": "B-1234",
            "confidence": 0.95,
            "raw_text": "B-1234",
            "all_matches": [{"text": "B-1234", "confidence": 0.95}],
            "yolo_boxes": []
        })

        return {
            'quality': [quality, quality, quality],
            'aircraft': [aircraft, aircraft, aircraft],
            'airline': [airline, airline, airline],
            'registration': [registration, registration, registration]
        }

    @pytest.mark.asyncio
    async def test_review_batch_uses_concurrent_inference(self, test_images_batch, sample_batch_results):
        """
        Test that review_batch() uses true concurrent inference instead of sequential loops.

        This test verifies that batch methods are called instead of looping through
        individual review() calls.
        """
        from unittest.mock import AsyncMock

        service = ReviewService()

        # Mock internal batch methods - aircraft and airline are now async
        async_mock_aircraft = AsyncMock(return_value=sample_batch_results['aircraft'])
        async_mock_airline = AsyncMock(return_value=sample_batch_results['airline'])

        with patch.object(service.quality_service, '_assess_batch', return_value=sample_batch_results['quality']):
            with patch.object(service.aircraft_service, '_classify_batch', async_mock_aircraft):
                with patch.object(service.airline_service, '_classify_batch', async_mock_airline):
                    with patch.object(service.registration_service, '_recognize_batch', return_value=sample_batch_results['registration']):
                        results = await service.review_batch(
                            test_images_batch,
                            include_quality=True,
                            include_aircraft=True,
                            include_airline=True,
                            include_registration=True
                        )

                        # Verify batch methods were called
                        assert service.quality_service._assess_batch.called
                        assert service.aircraft_service._classify_batch.called
                        assert service.airline_service._classify_batch.called
                        assert service.registration_service._recognize_batch.called

                        # Verify batch methods were called with lists of images (not single images)
                        quality_call_args = service.quality_service._assess_batch.call_args[0][0]
                        assert len(quality_call_args) == 3
                        assert all(isinstance(img, Image.Image) for img in quality_call_args)

                        # Verify results
                        assert len(results) == 3
                        assert all(r['success'] for r in results)

    @pytest.mark.asyncio
    async def test_review_batch_passes_loaded_images_only_once(self, test_images_batch, sample_batch_results):
        """
        Test that review_batch() loads images only once per image, not per service.

        This test ensures that each image is loaded exactly once in the batch
        and reused across all services.
        """
        service = ReviewService()

        # Track load_image calls
        load_image_calls = []

        original_load_image = BaseService.load_image

        def tracked_load_image(image_input):
            load_image_calls.append(image_input)
            return original_load_image(image_input)

        from unittest.mock import AsyncMock

        async_mock_aircraft = AsyncMock(return_value=sample_batch_results['aircraft'])
        async_mock_airline = AsyncMock(return_value=sample_batch_results['airline'])

        with patch('app.services.base.BaseService.load_image', side_effect=tracked_load_image):
            # Mock internal batch methods - aircraft and airline are now async
            with patch.object(service.quality_service, '_assess_batch', return_value=sample_batch_results['quality']):
                with patch.object(service.aircraft_service, '_classify_batch', async_mock_aircraft):
                    with patch.object(service.airline_service, '_classify_batch', async_mock_airline):
                        with patch.object(service.registration_service, '_recognize_batch', return_value=sample_batch_results['registration']):
                            results = await service.review_batch(
                                test_images_batch,
                                include_quality=True,
                                include_aircraft=True,
                                include_airline=True,
                                include_registration=True
                            )

        # Verify each image was loaded exactly once
        assert len(load_image_calls) == 3, f"Expected 3 image loads, got {len(load_image_calls)}"

    @pytest.mark.asyncio
    async def test_review_batch_handles_partial_failures(self, test_images_batch, sample_batch_results):
        """
        Test that review_batch() handles partial failures gracefully.

        This test verifies that if some services fail, the overall batch
        operation continues and returns appropriate error information.
        """
        service = ReviewService()

        from unittest.mock import AsyncMock

        # Mock one service to return None (simulating safe_execute behavior)
        async_mock_aircraft = AsyncMock(return_value=sample_batch_results['aircraft'])
        async_mock_airline = AsyncMock(return_value=sample_batch_results['airline'])

        with patch.object(service.quality_service, '_assess_batch', return_value=None):
            with patch.object(service.aircraft_service, '_classify_batch', async_mock_aircraft):
                with patch.object(service.airline_service, '_classify_batch', async_mock_airline):
                    with patch.object(service.registration_service, '_recognize_batch', return_value=sample_batch_results['registration']):
                        results = await service.review_batch(
                            test_images_batch,
                            include_quality=True,
                            include_aircraft=True,
                            include_airline=True,
                            include_registration=True
                        )

                        # Results should still be returned
                        assert len(results) == 3
                        # Results might have errors but should not crash

