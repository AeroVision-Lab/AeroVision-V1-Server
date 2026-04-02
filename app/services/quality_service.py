"""
Quality assessment service.

Supports three modes (configured via QUALITY_MODE env var):
- opencv: Fast classical CV metrics only
- qwen: Semantic VLM analysis only
- hybrid: Both in parallel (default)
"""

import asyncio
from typing import Any

from PIL import Image

from app.core import get_logger, get_settings
from app.schemas.quality import QualityResult
from app.services.base import BaseService
from app.services.quality import HybridAssessor, OpenCVDetector, QwenDetector
from app.services.quality.result_merger import merge_results

logger = get_logger("services.quality")


class QualityService(BaseService):
    """Service for image quality assessment with mode-based routing."""

    def __init__(self):
        """Initialize the quality service."""
        self._settings = get_settings()
        self._opencv_detector: OpenCVDetector | None = None
        self._qwen_detector: QwenDetector | None = None
        self._hybrid_assessor: HybridAssessor | None = None

    def _get_opencv_detector(self) -> OpenCVDetector:
        """Lazy load OpenCV detector."""
        if self._opencv_detector is None:
            self._opencv_detector = OpenCVDetector()
        return self._opencv_detector

    def _get_qwen_detector(self) -> QwenDetector:
        """Lazy load Qwen detector."""
        if self._qwen_detector is None:
            self._qwen_detector = QwenDetector()
        return self._qwen_detector

    def _get_hybrid_assessor(self) -> HybridAssessor:
        """Lazy load hybrid assessor."""
        if self._hybrid_assessor is None:
            self._hybrid_assessor = HybridAssessor()
        return self._hybrid_assessor

    async def assess(self, image_input: str) -> tuple[QualityResult, float]:
        """
        Assess image quality using the configured mode.

        Args:
            image_input: Base64 encoded image or URL

        Returns:
            Tuple of (quality result, processing time ms)
        """
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, self.load_image, image_input)
        return await self._assess_image(image)

    async def _assess_image(self, image: Image.Image) -> tuple[QualityResult, float]:
        """
        Assess quality of a pre-loaded image based on quality_mode.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (quality result, processing time ms)
        """
        mode = self._settings.quality_mode
        logger.info("Running quality assessment in '%s' mode", mode)

        async def do_assess() -> QualityResult:
            if mode == "opencv":
                # OpenCV only - run in thread pool
                loop = asyncio.get_event_loop()
                opencv_result = await loop.run_in_executor(
                    None, self._get_opencv_detector().detect, image
                )
                # Convert OpenCVResult to QualityResult
                return self._opencv_to_quality_result(opencv_result)

            elif mode == "qwen":
                # Qwen only
                qwen_result = await self._get_qwen_detector().detect(image)
                # Convert QwenResult to QualityResult
                return self._qwen_to_quality_result(qwen_result)

            else:  # hybrid
                # Both in parallel
                return await self._get_hybrid_assessor().assess(image)

        result, timing = await self.measure_time_async(do_assess)
        return result, timing

    def _opencv_to_quality_result(self, opencv_result) -> QualityResult:
        """Convert OpenCVResult to QualityResult (opencv-only mode)."""
        from app.services.quality.opencv_detector import OpenCVResult

        # Calculate overall score from OpenCV dimensions
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
        overall_score = sum(scores) / len(scores)

        # Determine pass/fail
        has_critical = any(v.severity == "critical" for v in opencv_result.violations)
        pass_ = overall_score >= self._settings.quality_pass_threshold and not has_critical

        return QualityResult.model_validate({
            "pass": pass_,
            "score": round(overall_score, 3),
            "violations": opencv_result.violations,
            "details": opencv_result.dimension_scores,
            "suggestions": [],
        })

    def _qwen_to_quality_result(self, qwen_result) -> QualityResult:
        """Convert QwenResult to QualityResult (qwen-only mode)."""
        from app.services.quality.qwen_detector import QwenResult

        # Calculate score based on violations
        score = 1.0
        for violation in qwen_result.violations:
            if violation.severity == "critical":
                score -= 0.30 * violation.confidence
            elif violation.severity == "major":
                score -= 0.15 * violation.confidence
            elif violation.severity == "minor":
                score -= 0.05 * violation.confidence
        score = max(0.0, score)

        # Determine pass/fail
        has_critical = any(v.severity == "critical" for v in qwen_result.violations)
        pass_ = score >= self._settings.quality_pass_threshold and not has_critical

        return QualityResult.model_validate({
            "pass": pass_,
            "score": round(score, 3),
            "violations": qwen_result.violations,
            "details": None,
            "suggestions": qwen_result.suggestions,
        })

    async def assess_batch(self, image_inputs: list[str]) -> list[dict[str, Any]]:
        """
        Assess quality of multiple images.

        Args:
            image_inputs: List of base64 encoded images or URLs

        Returns:
            List of results with index, success status, and data/error
        """
        # Load all images in parallel
        async def load_image_async(image_input: str):
            try:
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, self.load_image, image_input)
                return image
            except Exception:
                return None

        images = await asyncio.gather(*[load_image_async(img) for img in image_inputs])

        # Assess all images in parallel
        tasks = []
        for image in images:
            if image is None:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
            else:
                tasks.append(asyncio.create_task(self._assess_image(image)))

        quality_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format results
        results = []
        for idx, result in enumerate(quality_results):
            if result is None or isinstance(result, Exception):
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": "Quality assessment failed"
                })
            else:
                quality_result, _ = result  # unpack (QualityResult, timing)
                results.append({
                    "index": idx,
                    "success": True,
                    "data": quality_result.model_dump(by_alias=True),
                    "error": None
                })

        return results
