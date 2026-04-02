"""
Hybrid quality assessor that runs OpenCV and Qwen detectors in parallel.

Execution strategy:
- Run OpenCV (sync) and Qwen (async) concurrently via asyncio.gather
- Merge results using result_merger
- Return unified QualityResult
"""

from __future__ import annotations

import asyncio

from PIL import Image

from app.core import get_logger
from app.schemas.quality import QualityResult
from app.services.quality.opencv_detector import OpenCVDetector
from app.services.quality.qwen_detector import QwenDetector
from app.services.quality.result_merger import merge_results

logger = get_logger("services.quality.hybrid")


class HybridAssessor:
    """
    Runs both OpenCV and Qwen quality detectors in parallel and merges results.
    """

    def __init__(self) -> None:
        self._opencv_detector = OpenCVDetector()
        self._qwen_detector = QwenDetector()

    async def assess(self, image: Image.Image) -> QualityResult:
        """
        Run hybrid quality assessment (OpenCV + Qwen in parallel).

        Args:
            image: PIL Image to assess.

        Returns:
            QualityResult with merged violations and scores.
        """
        logger.info("Starting hybrid quality assessment")

        # Run both detectors in parallel
        opencv_task = asyncio.to_thread(self._opencv_detector.detect, image)
        qwen_task = self._qwen_detector.detect(image)

        opencv_result, qwen_result = await asyncio.gather(opencv_task, qwen_task)

        # Merge results
        final_result = merge_results(opencv_result, qwen_result)

        logger.info(
            "Hybrid assessment complete: score=%.2f pass=%s violations=%d",
            final_result.score,
            final_result.pass_,
            len(final_result.violations),
        )

        return final_result
