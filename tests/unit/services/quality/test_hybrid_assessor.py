"""
Tests for hybrid quality assessor.
"""

import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.quality.hybrid_assessor import HybridAssessor
from app.services.quality.opencv_detector import OpenCVResult
from app.services.quality.qwen_detector import QwenResult
from app.schemas.quality import RuleViolation


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return Image.new("RGB", (800, 600), color=(100, 150, 200))


@pytest.mark.asyncio
async def test_hybrid_assess_both_detectors(sample_image):
    """Test hybrid assessment runs both detectors."""
    with patch("app.services.quality.hybrid_assessor.OpenCVDetector") as MockOpenCV, \
         patch("app.services.quality.hybrid_assessor.QwenDetector") as MockQwen:

        # Mock OpenCV result
        opencv_result = OpenCVResult()
        opencv_result.sharpness_score = 0.8
        opencv_result.violations = []

        # Mock Qwen result
        qwen_result = QwenResult(
            violations=[],
            suggestions=["建议改进"],
            service_name="primary"
        )

        mock_opencv = MockOpenCV.return_value
        mock_opencv.detect = MagicMock(return_value=opencv_result)

        mock_qwen = MockQwen.return_value
        mock_qwen.detect = AsyncMock(return_value=qwen_result)

        assessor = HybridAssessor()
        result = await assessor.assess(sample_image)

        # Verify both detectors were called
        mock_opencv.detect.assert_called_once()
        mock_qwen.detect.assert_called_once()

        # Verify result structure
        assert result.pass_ is not None
        assert 0 <= result.score <= 1
        assert isinstance(result.violations, list)


@pytest.mark.asyncio
async def test_hybrid_assess_merges_violations(sample_image):
    """Test that violations from both sources are merged."""
    with patch("app.services.quality.hybrid_assessor.OpenCVDetector") as MockOpenCV, \
         patch("app.services.quality.hybrid_assessor.QwenDetector") as MockQwen:

        opencv_result = OpenCVResult()
        opencv_result.violations = [
            RuleViolation(
                rule_id="Rules1.1.1",
                rule_name="模糊",
                severity="minor",
                confidence=0.3,
                description="轻微模糊",
                source="opencv"
            )
        ]

        qwen_result = QwenResult(
            violations=[
                RuleViolation(
                    rule_id="Rules1.2.2",
                    rule_name="反光",
                    severity="major",
                    confidence=0.8,
                    description="明显反光",
                    source="qwen"
                )
            ],
            suggestions=[],
            service_name="primary"
        )

        mock_opencv = MockOpenCV.return_value
        mock_opencv.detect = MagicMock(return_value=opencv_result)

        mock_qwen = MockQwen.return_value
        mock_qwen.detect = AsyncMock(return_value=qwen_result)

        assessor = HybridAssessor()
        result = await assessor.assess(sample_image)

        # Should have violations from both sources
        assert len(result.violations) == 2
        sources = {v.source for v in result.violations}
        assert sources == {"opencv", "qwen"}
