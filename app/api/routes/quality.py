"""
Quality assessment API endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.api.deps import increment_request_count
from app.schemas.common import ImageInput, BatchImageInput, Meta
from app.schemas.quality import (
    QualityResponse,
    BatchQualityResponse,
    BatchQualityItem,
)
from app.services import QualityService

router = APIRouter(prefix="/quality", tags=["Quality"])
_service = QualityService()


@router.post("", response_model=QualityResponse, status_code=status.HTTP_200_OK)
async def assess_quality(request: ImageInput) -> QualityResponse:
    """
    Assess image quality.

    Evaluates sharpness, exposure, composition, noise, and color.
    Returns an overall score and detailed metrics.
    """
    try:
        result, timing_ms = _service.assess(request.image)
        increment_request_count(success=True)

        return QualityResponse(
            **result.model_dump(by_alias=True),
            meta=Meta(processing_time_ms=timing_ms)
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality assessment failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchQualityResponse, status_code=status.HTTP_200_OK)
async def assess_quality_batch(request: BatchImageInput) -> BatchQualityResponse:
    """
    Batch assess image quality.

    Evaluates quality for up to 50 images in a single request.
    """
    try:
        start_time = datetime.utcnow()
        service_results = await _service.assess_batch(request.images)

        results = [
            BatchQualityItem(**r)
            for r in service_results
        ]

        successful = sum(1 for r in service_results if r["success"])
        failed = len(service_results) - successful

        increment_request_count(success=True)

        return BatchQualityResponse(
            total=len(service_results),
            successful=successful,
            failed=failed,
            results=results
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch quality assessment failed: {str(e)}"
        )
