"""
Airline classification service.
"""

from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.inference import InferenceFactory, wrap_airline_result
from app.schemas.airline import AirlineResult
from app.services.base import BaseService
from app.services._classifier_base import ClassificationServiceBase


class AirlineService(ClassificationServiceBase):
    """Service for airline classification."""

    def __init__(self):
        """Initialize the airline service."""
        super().__init__(
            classifier_factory=InferenceFactory.get_airline_classifier,
            result_wrapper=wrap_airline_result,
            result_type=AirlineResult,
            service_name="airline_service",
            error_message="Airline classification failed"
        )
