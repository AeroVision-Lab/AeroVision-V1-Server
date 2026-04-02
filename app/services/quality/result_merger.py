"""
Result merger for combining OpenCV and Qwen detection results.

Merging strategy:
- Violations: combine both sources, Qwen takes priority on overlapping rules
- Score: weighted average (OpenCV 30% + Qwen semantic 70%)
- Pass/fail: based on final score and violation severity
- Suggestions: combine from both sources
"""

from __future__ import annotations

from app.core import get_logger
from app.schemas.quality import QualityDetails, QualityResult, RuleViolation
from app.services.quality.opencv_detector import OpenCVResult
from app.services.quality.qwen_detector import QwenResult

logger = get_logger("services.quality.merger")


# ---------------------------------------------------------------------------
# Scoring weights and thresholds
# ---------------------------------------------------------------------------

# Overall score = OpenCV_score * 0.3 + Qwen_score * 0.7
_OPENCV_WEIGHT = 0.30
_QWEN_WEIGHT = 0.70

# Pass threshold: score >= 0.60 AND no critical violations
_PASS_SCORE_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Main merger function
# ---------------------------------------------------------------------------

def merge_results(
    opencv_result: OpenCVResult,
    qwen_result: QwenResult,
) -> QualityResult:
    """
    Merge OpenCV and Qwen results into a unified QualityResult.

    Args:
        opencv_result: OpenCV detection result.
        qwen_result: Qwen detection result.

    Returns:
        QualityResult with combined violations, score, and pass/fail.
    """
    # --- Combine violations (no deduplication needed - different rule sets) ---
    all_violations = opencv_result.violations + qwen_result.violations

    # --- Calculate OpenCV score (average of all dimension scores) ---
    opencv_score = _calculate_opencv_score(opencv_result)

    # --- Calculate Qwen score (based on violations) ---
    qwen_score = _calculate_qwen_score(qwen_result)

    # --- Weighted overall score ---
    overall_score = opencv_score * _OPENCV_WEIGHT + qwen_score * _QWEN_WEIGHT

    # --- Determine pass/fail ---
    has_critical = any(v.severity == "critical" for v in all_violations)
    pass_ = overall_score >= _PASS_SCORE_THRESHOLD and not has_critical

    # --- Combine suggestions ---
    suggestions = qwen_result.suggestions  # Qwen provides semantic suggestions

    logger.info(
        "Merged results: opencv_score=%.2f qwen_score=%.2f overall=%.2f "
        "violations=%d pass=%s",
        opencv_score, qwen_score, overall_score, len(all_violations), pass_,
    )

    return QualityResult.model_validate({
        "pass": pass_,
        "score": round(overall_score, 3),
        "violations": all_violations,
        "details": opencv_result.dimension_scores,
        "suggestions": suggestions,
    })


# ---------------------------------------------------------------------------
# Score calculation helpers
# ---------------------------------------------------------------------------

def _calculate_opencv_score(opencv_result: OpenCVResult) -> float:
    """
    Calculate overall OpenCV score as average of all dimension scores.

    Args:
        opencv_result: OpenCV detection result.

    Returns:
        Average score in [0, 1].
    """
    scores = [
        opencv_result.sharpness_score,
        opencv_result.noise_score,
        opencv_result.contrast_score,
        opencv_result.exposure_score,
        opencv_result.color_score,
        opencv_result.horizon_score,
        opencv_result.vignetting_score,
        opencv_result.haze_score,
        opencv_result.purple_fringe_score,
    ]
    return sum(scores) / len(scores)


def _calculate_qwen_score(qwen_result: QwenResult) -> float:
    """
    Calculate Qwen score based on violations.

    Scoring logic:
    - Start at 1.0 (perfect)
    - Deduct for each violation based on severity and confidence:
      - critical: -0.30 * confidence
      - major: -0.15 * confidence
      - minor: -0.05 * confidence
    - Floor at 0.0

    Args:
        qwen_result: Qwen detection result.

    Returns:
        Score in [0, 1].
    """
    score = 1.0

    for violation in qwen_result.violations:
        if violation.severity == "critical":
            score -= 0.30 * violation.confidence
        elif violation.severity == "major":
            score -= 0.15 * violation.confidence
        elif violation.severity == "minor":
            score -= 0.05 * violation.confidence

    return max(0.0, score)