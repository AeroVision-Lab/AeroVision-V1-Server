"""
Aircraft type classification API endpoints.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import increment_request_count
from app.core.exceptions import ImageLoadError, InferenceError
from app.schemas.common import ImageInput, BatchImageInput, Meta
from app.schemas.aircraft import (
    AircraftResponse,
    BatchAircraftResponse,
    BatchAircraftItem,
)
from app.services import AircraftService

router = APIRouter(prefix="/aircraft", tags=["Aircraft"])
_service = AircraftService()


@router.post("", response_model=AircraftResponse, status_code=status.HTTP_200_OK)
async def classify_aircraft(
    request: ImageInput,
    top_k: Annotated[int | None, Query(gt=0, le=20, description="Number of top predictions")] = None
) -> AircraftResponse:
    """
    Classify aircraft type.

    Returns the predicted aircraft type with confidence score.
    Optionally returns top-k predictions.
    """
    try:
        result, timing_ms = _service.classify(request.image, top_k=top_k)
        increment_request_count(success=True)

        return AircraftResponse(
            **result.model_dump(by_alias=True),
            meta=Meta(processing_time_ms=timing_ms)
        )
    except ImageLoadError as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load image: {str(e)}"
        )
    except (InferenceError, Exception) as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aircraft classification failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchAircraftResponse, status_code=status.HTTP_200_OK)
async def classify_aircraft_batch(
    request: BatchImageInput,
    top_k: Annotated[int | None, Query(gt=0, le=20, description="Number of top predictions")] = None
) -> BatchAircraftResponse:
    """
    Batch classify aircraft types.

    Classifies up to 50 images in a single request.
    """
    try:
        service_results = await _service.classify_batch(request.images, top_k=top_k)

        results = [BatchAircraftItem(**r) for r in service_results]

        successful = sum(1 for r in service_results if r["success"])
        failed = len(service_results) - successful

        increment_request_count(success=True)

        return BatchAircraftResponse(
            total=len(service_results),
            successful=successful,
            failed=failed,
            results=results
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch aircraft classification failed: {str(e)}"
        )
