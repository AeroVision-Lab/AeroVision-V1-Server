"""
Pydantic schemas for Aerovision-V1-Server.
"""

from app.schemas.aircraft import (
    AircraftResponse,
    AircraftResult,
    BatchAircraftItem,
    BatchAircraftResponse,
    Prediction,
)
from app.schemas.airline import (
    AirlineResponse,
    AirlineResult,
    BatchAirlineItem,
    BatchAirlineResponse,
)
from app.schemas.common import (
    BatchImageInput,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ImageInput,
    Meta,
    StatsResponse,
    SuccessResponse,
)
from app.schemas.quality import (
    QualityResponse,
    QualityResult,
    QualityDetails,
    BatchQualityItem,
    BatchQualityResponse,
)
from app.schemas.registration import (
    RegistrationResponse,
    RegistrationResult,
    BatchRegistrationItem,
    BatchRegistrationResponse,
    YoloBox,
    OcrMatch,
)
from app.schemas.review import (
    ReviewResponse,
    ReviewResult,
    BatchReviewResponse,
    BatchReviewItem,
    ReviewQualityResult,
    ReviewAircraftResult,
    ReviewAirlineResult,
    ReviewRegistrationResult,
)

__all__ = [
    # Common
    "ImageInput",
    "BatchImageInput",
    "Meta",
    "SuccessResponse",
    "ErrorResponse",
    "ErrorDetail",
    "HealthResponse",
    "StatsResponse",
    # Quality
    "QualityResponse",
    "QualityResult",
    "QualityDetails",
    "BatchQualityItem",
    "BatchQualityResponse",
    # Aircraft
    "AircraftResponse",
    "AircraftResult",
    "BatchAircraftItem",
    "BatchAircraftResponse",
    "Prediction",
    # Airline
    "AirlineResponse",
    "AirlineResult",
    "BatchAirlineItem",
    "BatchAirlineResponse",
    # Registration
    "RegistrationResponse",
    "RegistrationResult",
    "BatchRegistrationItem",
    "BatchRegistrationResponse",
    "YoloBox",
    "OcrMatch",
    # Review
    "ReviewResponse",
    "ReviewResult",
    "BatchReviewResponse",
    "BatchReviewItem",
    "ReviewQualityResult",
    "ReviewAircraftResult",
    "ReviewAirlineResult",
    "ReviewRegistrationResult",
]
