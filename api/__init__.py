"""
API 模块
"""

from fastapi import APIRouter

from .routes import review, health, history

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(review.router, prefix="/review", tags=["review"])
api_router.include_router(history.router, prefix="/history", tags=["history"])

__all__ = ["api_router"]
