"""
Aggregated review API endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import increment_request_count
from app.schemas.common import ImageInput, BatchImageInput, Meta
from app.schemas.review import (
    ReviewResponse,
    BatchReviewResponse,
    BatchReviewItem,
)
from app.services import ReviewService

router = APIRouter(prefix="/review", tags=["Review"])
_service = ReviewService()


@router.post("", response_model=ReviewResponse, status_code=status.HTTP_200_OK)
async def review_image(
    request: ImageInput,
    include_quality: Annotated[bool, Query(description="Include quality assessment")] = True,
    include_aircraft: Annotated[bool, Query(description="Include aircraft classification")] = True,
    include_airline: Annotated[bool, Query(description="Include airline classification")] = True,
    include_registration: Annotated[bool, Query(description="Include registration OCR")] = True
) -> ReviewResponse:
    """
    Perform a complete review of an image.

    Aggregates quality assessment, aircraft/airline classification, and registration OCR.
    Each component can be toggled via query parameters.
    """
    try:
        result, timing_ms = _service.review(
            request.image,
            include_quality=include_quality,
            include_aircraft=include_aircraft,
            include_airline=include_airline,
            include_registration=include_registration
        )
        increment_request_count(success=True)

        return ReviewResponse(
            **result.model_dump(by_alias=True),
            meta=Meta(processing_time_ms=timing_ms)
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Review failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchReviewResponse, status_code=status.HTTP_200_OK)
async def review_batch(
    request: BatchImageInput,
    include_quality: Annotated[bool, Query(description="Include quality assessment")] = True,
    include_aircraft: Annotated[bool, Query(description="Include aircraft classification")] = True,
    include_airline: Annotated[bool, Query(description="Include airline classification")] = True,
    include_registration: Annotated[bool, Query(description="Include registration OCR")] = True
) -> BatchReviewResponse:
    """
    Batch review images.

    Performs complete review on up to 50 images in a single request.
    Each component can be toggled via query parameters.
    """
    try:
        service_results = await _service.review_batch(
            request.images,
            include_quality=include_quality,
            include_aircraft=include_aircraft,
            include_airline=include_airline,
            include_registration=include_registration
        )

        results = [BatchReviewItem(**r) for r in service_results]

        successful = sum(1 for r in service_results if r["success"])
        failed = len(service_results) - successful

        increment_request_count(success=True)

        return BatchReviewResponse(
            total=len(service_results),
            successful=successful,
            failed=failed,
            results=results
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch review failed: {str(e)}"
        )
