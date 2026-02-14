"""
Aircraft type classification service.
"""

from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.inference import InferenceFactory, wrap_aircraft_result
from app.schemas.aircraft import AircraftResult
from app.services.base import BaseService
from app.services._classifier_base import ClassificationServiceBase


class AircraftService(ClassificationServiceBase):
    """Service for aircraft type classification."""

    def __init__(self):
        """Initialize the aircraft service."""
        super().__init__(
            classifier_factory=InferenceFactory.get_aircraft_classifier,
            result_wrapper=wrap_aircraft_result,
            result_type=AircraftResult,
            service_name="aircraft_service",
            error_message="Aircraft classification failed"
        )
