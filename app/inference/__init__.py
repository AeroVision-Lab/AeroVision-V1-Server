"""
Inference layer for Aerovision-V1-Server.

This module provides the interface to aerovision_inference models.
"""

from app.inference.factory import (
    InferenceFactory,
    InferenceFactoryError,
    INFERENCE_AVAILABLE,
)
from app.inference.wrappers import (
    wrap_quality_result,
    wrap_aircraft_result,
    wrap_airline_result,
    wrap_registration_result,
)

__all__ = [
    "InferenceFactory",
    "InferenceFactoryError",
    "INFERENCE_AVAILABLE",
    "wrap_quality_result",
    "wrap_aircraft_result",
    "wrap_airline_result",
    "wrap_registration_result",
]
