"""
Quality assessment sub-package.

Provides modular detectors used by the hybrid quality assessor.
"""

from app.services.quality.opencv_detector import OpenCVDetector, OpenCVResult

__all__ = ["OpenCVDetector", "OpenCVResult"]
