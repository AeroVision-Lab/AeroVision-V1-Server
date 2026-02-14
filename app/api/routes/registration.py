"""
Registration number OCR API endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import increment_request_count
from app.schemas.common import ImageInput, BatchImageInput, Meta
from app.schemas.registration import (
    RegistrationResponse,
    BatchRegistrationResponse,
    BatchRegistrationItem,
)
from app.services import RegistrationService

router = APIRouter(prefix="/registration", tags=["Registration"])
_service = RegistrationService()


@router.post("", response_model=RegistrationResponse, status_code=status.HTTP_200_OK)
async def recognize_registration(request: ImageInput) -> RegistrationResponse:
    """
    Recognize aircraft registration number.

    Performs full registration detection and OCR.
    Returns the recognized registration number with confidence and bounding boxes.
    """
    try:
        result, timing_ms = _service.recognize(request.image)
        increment_request_count(success=True)

        return RegistrationResponse(
            **result.model_dump(by_alias=True),
            meta=Meta(processing_time_ms=timing_ms)
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration OCR failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchRegistrationResponse, status_code=status.HTTP_200_OK)
async def recognize_registration_batch(request: BatchImageInput) -> BatchRegistrationResponse:
    """
    Batch recognize registration numbers.

    Recognizes registration numbers from up to 50 images in a single request.
    """
    try:
        service_results = await _service.recognize_batch(request.images)

        results = [BatchRegistrationItem(**r) for r in service_results]

        successful = sum(1 for r in service_results if r["success"])
        failed = len(service_results) - successful

        increment_request_count(success=True)

        return BatchRegistrationResponse(
            total=len(service_results),
            successful=successful,
            failed=failed,
            results=results
        )
    except Exception as e:
        increment_request_count(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch registration OCR failed: {str(e)}"
        )
