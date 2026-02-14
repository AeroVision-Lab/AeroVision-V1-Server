"""
Airline classification API endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import increment_request_count
from app.schemas.common import ImageInput, BatchImageInput, Meta
from app.schemas.airline import (
    AirlineResponse,
    BatchAirlineResponse,
    BatchAirlineItem,
)
from app.services import AirlineService

router = APIRouter(prefix="/airline", tags=["Airline"])
_service = AirlineService()


@router.post("", response_model=AirlineResponse, status_code=status.HTTP_200_OK)
async def classify_airline(
    request: ImageInput,
    top_k: Annotated[int | None, Query(gt=0, le=20, description="Number of top predictions")] = None
) -> AirlineResponse:
    """
    Classify airline.

    Returns the predicted airline code with confidence score.
    Optionally returns top-k predictions.
    """
    try:
        result, timing_ms = _service.classify(request.image, top_k=top_k)
        increment_request_count(success=True)

        return AirlineResponse(
            **result.model_dump(by_alias=True),
            meta=Meta(processing_time_ms=timing_ms)
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Airline classification failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchAirlineResponse, status_code=status.HTTP_200_OK)
async def classify_airline_batch(
    request: BatchImageInput,
    top_k: Annotated[int | None, Query(gt=0, le=20, description="Number of top predictions")] = None
) -> BatchAirlineResponse:
    """
    Batch classify airlines.

    Classifies up to 50 images in a single request.
    """
    try:
        service_results = await _service.classify_batch(request.images, top_k=top_k)

        results = [BatchAirlineItem(**r) for r in service_results]

        successful = sum(1 for r in service_results if r["success"])
        failed = len(service_results) - successful

        increment_request_count(success=True)

        return BatchAirlineResponse(
            total=len(service_results),
            successful=successful,
            failed=failed,
            results=results
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch airline classification failed: {str(e)}"
        )
