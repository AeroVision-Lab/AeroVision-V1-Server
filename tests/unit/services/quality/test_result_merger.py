"""
Tests for result merger.
"""

import pytest
from PIL import Image

from app.schemas.quality import RuleViolation
from app.services.quality.opencv_detector import OpenCVDetector, OpenCVResult
from app.services.quality.qwen_detector import QwenResult
from app.services.quality.result_merger import merge_results


@pytest.fixture
def opencv_result():
    """Create a sample OpenCV result."""
    result = OpenCVResult()
    result.sharpness_score = 0.8
    result.noise_score = 0.9
    result.contrast_score = 0.7
    result.exposure_score = 0.85
    result.color_score = 0.75
    result.horizon_score = 0.95
    result.vignetting_score = 0.8
    result.haze_score = 0.9
    result.purple_fringe_score = 0.85
    result.violations = [
        RuleViolation(
            rule_id="Rules1.1.1",
            rule_name="模糊/虚焦",
            severity="minor",
            confidence=0.2,
            description="轻微模糊",
            source="opencv"
        )
    ]
    return result


@pytest.fixture
def qwen_result():
    """Create a sample Qwen result."""
    return QwenResult(
        violations=[
            RuleViolation(
                rule_id="Rules1.2.2",
                rule_name="玻璃反光",
                severity="major",
                confidence=0.85,
                description="明显反光",
                source="qwen"
            )
        ],
        suggestions=["避免玻璃窗拍摄"],
        service_name="primary"
    )


def test_merge_results_basic(opencv_result, qwen_result):
    """Test basic result merging."""
    result = merge_results(opencv_result, qwen_result)

    assert result.pass_ is True  # score should be above threshold
    assert 0 <= result.score <= 1
    assert len(result.violations) == 2  # 1 from OpenCV + 1 from Qwen
    assert len(result.suggestions) == 1
    assert result.details is not None  # OpenCV dimension scores preserved


def test_merge_results_critical_violation(opencv_result, qwen_result):
    """Test merging with critical violation."""
    qwen_result.violations[0].severity = "critical"
    result = merge_results(opencv_result, qwen_result)

    assert result.pass_ is False  # critical violation should fail
    assert len(result.violations) == 2


def test_merge_results_low_score(opencv_result, qwen_result):
    """Test merging with low scores."""
    # Make OpenCV scores very low
    opencv_result.sharpness_score = 0.1
    opencv_result.noise_score = 0.1
    opencv_result.contrast_score = 0.1
    opencv_result.exposure_score = 0.1
    opencv_result.color_score = 0.1
    opencv_result.horizon_score = 0.1
    opencv_result.vignetting_score = 0.1
    opencv_result.haze_score = 0.1
    opencv_result.purple_fringe_score = 0.1

    result = merge_results(opencv_result, qwen_result)

    assert result.pass_ is False  # low score should fail
    assert result.score < 0.6
