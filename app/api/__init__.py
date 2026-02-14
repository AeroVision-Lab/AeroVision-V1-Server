"""
API routes aggregation.
"""

from fastapi import APIRouter

from app.api.routes import health, quality, aircraft, airline, registration, review, history


api_router = APIRouter(prefix="/api/v1")

# Include all route modules
api_router.include_router(health.router)
api_router.include_router(quality.router)
api_router.include_router(aircraft.router)
api_router.include_router(airline.router)
api_router.include_router(registration.router)
api_router.include_router(review.router)
api_router.include_router(history.router, prefix="/history", tags=["history"])


__all__ = ["api_router"]
