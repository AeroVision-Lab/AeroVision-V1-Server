"""
Unit tests for OpenCV detector.
"""

import numpy as np
import pytest
from PIL import Image

from app.services.quality.opencv_detector import OpenCVDetector


class TestOpenCVDetector:
    """Test suite for OpenCVDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return OpenCVDetector()

    @pytest.fixture
    def clear_image(self):
        """Create a clear, well-exposed test image."""
        # 640x480 RGB image with good contrast and color
        arr = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @pytest.fixture
    def blurry_image(self):
        """Create a blurry test image."""
        arr = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add minimal variation (simulates blur)
        arr += np.random.randint(-5, 5, (480, 640, 3), dtype=np.int16).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @pytest.fixture
    def dark_image(self):
        """Create an underexposed test image (pixels mostly 0-14)."""
        arr = np.random.randint(0, 15, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    @pytest.fixture
    def bright_image(self):
        """Create an overexposed test image (pixels mostly 241-255)."""
        arr = np.random.randint(241, 256, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_detector_initialization(self, detector):
        """Test that detector can be instantiated."""
        assert detector is not None

    def test_detect_clear_image(self, detector, clear_image):
        """Test detection on a clear image."""
        result = detector.detect(clear_image)

        assert result is not None
        assert result.dimension_scores is not None
        assert 0.0 <= result.sharpness_score <= 1.0
        assert 0.0 <= result.noise_score <= 1.0
        assert 0.0 <= result.contrast_score <= 1.0
        assert 0.0 <= result.exposure_score <= 1.0
        assert 0.0 <= result.color_score <= 1.0

        # Clear image should have relatively few violations
        assert len(result.violations) < 5

    def test_detect_blurry_image(self, detector, blurry_image):
        """Test detection on a blurry image."""
        result = detector.detect(blurry_image)

        # Blurry image should have low sharpness score
        assert result.sharpness_score < 0.5

        # Should have at least one violation for blur
        violation_ids = [v.rule_id for v in result.violations]
        assert "Rules1.1.1" in violation_ids

    def test_detect_dark_image(self, detector, dark_image):
        """Test detection on an underexposed image."""
        result = detector.detect(dark_image)

        # Dark image should have low exposure score
        assert result.exposure_score < 0.5

        # Should have exposure violation
        violation_ids = [v.rule_id for v in result.violations]
        assert "Rules1.2.3" in violation_ids

    def test_detect_bright_image(self, detector, bright_image):
        """Test detection on an overexposed image."""
        result = detector.detect(bright_image)

        # Bright image should have low exposure score
        assert result.exposure_score < 0.5

        # Should have exposure violation
        violation_ids = [v.rule_id for v in result.violations]
        assert "Rules1.2.3" in violation_ids

    def test_violations_have_required_fields(self, detector, clear_image):
        """Test that all violations have required fields."""
        result = detector.detect(clear_image)

        for violation in result.violations:
            assert violation.rule_id
            assert violation.rule_name
            assert violation.severity in ["critical", "major", "minor"]
            assert 0.0 <= violation.confidence <= 1.0
            assert violation.description
            assert violation.source == "opencv"

    def test_dimension_scores_backward_compat(self, detector, clear_image):
        """Test that dimension_scores is populated for backward compatibility."""
        result = detector.detect(clear_image)

        assert result.dimension_scores is not None
        assert hasattr(result.dimension_scores, "sharpness")
        assert hasattr(result.dimension_scores, "exposure")
        assert hasattr(result.dimension_scores, "composition")
        assert hasattr(result.dimension_scores, "noise")
        assert hasattr(result.dimension_scores, "color")
