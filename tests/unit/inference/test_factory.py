"""
Unit tests for InferenceFactory.
"""

import os
import threading
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core.config import reload_settings
from app.inference.factory import (
    InferenceFactory,
    InferenceFactoryError,
    INFERENCE_AVAILABLE,
)


class TestInferenceFactory:
    """Tests for InferenceFactory."""

    @pytest.fixture(autouse=True)
    def reset_settings_after_each_test(self):
        """Reset settings after each test for isolation."""
        # Store original env vars
        original_model_dir = os.environ.get("MODEL_DIR")
        original_device = os.environ.get("DEVICE")

        yield

        # Restore original env vars and reload settings
        if original_model_dir is not None:
            os.environ["MODEL_DIR"] = original_model_dir
        elif "MODEL_DIR" in os.environ:
            del os.environ["MODEL_DIR"]

        if original_device is not None:
            os.environ["DEVICE"] = original_device
        elif "DEVICE" in os.environ:
            del os.environ["DEVICE"]

        reload_settings()

    def test_is_available(self):
        """Test is_available returns correct status."""
        # Just check the method exists and returns a bool
        assert isinstance(InferenceFactory.is_available(), bool)

    def test_get_model_dir_default(self):
        """Test get_model_dir returns default when not set."""
        # Unset env var if set
        os.environ.pop("MODEL_DIR", None)
        reload_settings()  # Clear cached settings
        model_dir = InferenceFactory.get_model_dir()
        assert str(model_dir) == "models"

    def test_get_model_dir_from_env(self):
        """Test get_model_dir reads from environment."""
        with patch.dict(os.environ, {"MODEL_DIR": "custom_models"}, clear=False):
            reload_settings()  # Clear cached settings
            model_dir = InferenceFactory.get_model_dir()
            assert str(model_dir) == "custom_models"

    def test_get_device_default(self):
        """Test get_device returns default when not set."""
        os.environ.pop("DEVICE", None)
        reload_settings()  # Clear cached settings
        device = InferenceFactory.get_device()
        assert device == "cuda"

    def test_get_device_from_env(self):
        """Test get_device reads from environment."""
        with patch.dict(os.environ, {"DEVICE": "cpu"}, clear=False):
            reload_settings()  # Clear cached settings
            device = InferenceFactory.get_device()
            assert device == "cpu"

    def test_reset(self):
        """Test reset clears cached instances."""
        # Reset should not raise any errors
        InferenceFactory.reset()
        # After reset, cached instances should be None
        assert InferenceFactory._aircraft_classifier is None
        assert InferenceFactory._airline_classifier is None
        assert InferenceFactory._registration_ocr is None
        assert InferenceFactory._quality_assessor is None

    @patch("app.inference.factory.INFERENCE_AVAILABLE", False)
    def test_get_aircraft_classifier_when_not_available(self):
        """Test error when inference package not available."""
        InferenceFactory.reset()

        with pytest.raises(InferenceFactoryError) as exc_info:
            InferenceFactory.get_aircraft_classifier()

        assert "not available" in str(exc_info.value)

    @patch("app.inference.factory.INFERENCE_AVAILABLE", False)
    def test_preload_models_when_not_available(self):
        """Test preload_models handles unavailability gracefully."""
        InferenceFactory.reset()
        # Should not raise, just log warning
        InferenceFactory.preload_models()

    @pytest.mark.skipif(not INFERENCE_AVAILABLE, reason="Inference package not available")
    def test_get_aircraft_classifier_thread_safety(self):
        """Test thread safety of get_aircraft_classifier with concurrent access."""
        InferenceFactory.reset()
        results = []
        errors = []

        def get_classifier():
            try:
                result = InferenceFactory.get_aircraft_classifier()
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads to access get_aircraft_classifier concurrently
        num_threads = 10
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=get_classifier)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all threads got the same instance (singleton)
        assert len(results) == num_threads
        first_result = results[0]
        for result in results[1:]:
            assert result is first_result, "All threads should receive the same instance"

    @pytest.mark.skipif(not INFERENCE_AVAILABLE, reason="Inference package not available")
    def test_get_airline_classifier_thread_safety(self):
        """Test thread safety of get_airline_classifier with concurrent access."""
        InferenceFactory.reset()
        results = []
        errors = []

        def get_classifier():
            try:
                result = InferenceFactory.get_airline_classifier()
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads to access get_airline_classifier concurrently
        num_threads = 10
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=get_classifier)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all threads got the same instance (singleton)
        assert len(results) == num_threads
        first_result = results[0]
        for result in results[1:]:
            assert result is first_result, "All threads should receive the same instance"

    @pytest.mark.skipif(not INFERENCE_AVAILABLE, reason="Inference package not available")
    def test_multiple_classifiers_concurrent_access(self):
        """Test thread safety when accessing multiple classifiers concurrently."""
        InferenceFactory.reset()
        results = {'aircraft': [], 'airline': [], 'ocr': [], 'quality': []}
        errors = []

        def get_all_classifiers():
            try:
                aircraft = InferenceFactory.get_aircraft_classifier()
                results['aircraft'].append(aircraft)

                airline = InferenceFactory.get_airline_classifier()
                results['airline'].append(airline)

                ocr = InferenceFactory.get_registration_ocr()
                results['ocr'].append(ocr)

                quality = InferenceFactory.get_quality_assessor()
                results['quality'].append(quality)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        num_threads = 10
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=get_all_classifiers)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify singleton pattern for each classifier
        for classifier_type, classifier_results in results.items():
            if classifier_results:  # Only check if results exist
                assert len(classifier_results) == num_threads
                first_result = classifier_results[0]
                for result in classifier_results[1:]:
                    assert result is first_result, f"{classifier_type} should return same instance"
