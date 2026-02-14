"""
Dependency injection for API routes.
"""

from typing import AsyncGenerator

from fastapi import Depends

from app.core.config import get_settings
from app.core.logging import logger
from app.core.redis_client import get_request_stats as get_redis_stats
from app.core.redis_client import increment_request_count as redis_increment


async def get_request_stats() -> dict:
    """
    Get current request statistics.

    Uses Redis for shared statistics across multiple workers.
    Falls back to local stats if Redis is unavailable.
    """
    return await get_redis_stats()


def increment_request_count(success: bool = True) -> None:
    """
    Increment request counters.

    Uses Redis for shared statistics across multiple workers.
    Falls back to local stats if Redis is unavailable.

    Args:
        success: Whether the request was successful
    """
    redis_increment(success=success)
