"""
Quality assessment sub-package.

Provides modular detectors used by the hybrid quality assessor.
"""

from app.services.quality.hybrid_assessor import HybridAssessor
from app.services.quality.opencv_detector import OpenCVDetector, OpenCVResult
from app.services.quality.qwen_detector import QwenDetector, QwenResult
from app.services.quality.result_merger import merge_results

__all__ = [
    "OpenCVDetector",
    "OpenCVResult",
    "QwenDetector",
    "QwenResult",
    "HybridAssessor",
    "merge_results",
]
