"""
Health check and statistics endpoints.
"""

import time
import torch

from fastapi import APIRouter, Depends

from app.api.deps import get_request_stats
from app.core.config import get_settings
from app.inference import InferenceFactory
from app.schemas.common import HealthResponse, StatsResponse

router = APIRouter(tags=["Health"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, version, and availability of inference models.
    """
    settings = get_settings()

    # Check GPU availability
    gpu_available = torch.cuda.is_available()

    # Check if inference models are loaded
    models_loaded = InferenceFactory.is_available()

    return HealthResponse(
        status="healthy",
        version=settings.version,
        models_loaded=models_loaded,
        gpu_available=gpu_available,
        uptime_seconds=time.time() - _start_time
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    stats: dict = Depends(get_request_stats)
) -> StatsResponse:
    """
    Get request statistics.

    Returns total requests, success/failure counts, and throughput metrics.
    """
    return StatsResponse(**stats)
