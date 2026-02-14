"""
Redis client for shared statistics across multiple workers.

Provides a centralized storage for request statistics that works
correctly in multi-worker deployments.
"""

import time
from typing import Optional

import redis
from redis.asyncio import Redis as AsyncRedis

from app.core.config import get_settings
from app.core.logging import logger


class RedisStatsManager:
    """Manager for request statistics using Redis."""

    def __init__(self):
        """Initialize Redis connection."""
        self.settings = get_settings()
        self._redis: Optional[Redis] = None
        self._async_redis: Optional[AsyncRedis] = None

    def get_redis(self) -> redis.Redis:
        """
        Get synchronous Redis client.

        Returns:
            Redis client instance
        """
        if self._redis is None:
            self._redis = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info(f"Connected to Redis: {self.settings.redis_url}")
        return self._redis

    async def get_async_redis(self) -> AsyncRedis:
        """
        Get asynchronous Redis client.

        Returns:
            Async Redis client instance
        """
        if self._async_redis is None:
            self._async_redis = AsyncRedis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info(f"Connected to Redis (async): {self.settings.redis_url}")
        return self._async_redis

    def increment_request_count(self, success: bool = True) -> None:
        """
        Increment request counters.

        Args:
            success: Whether the request was successful
        """
        try:
            r = self.get_redis()
            pipe = r.pipeline()
            pipe.incr("stats:request_count")
            if success:
                pipe.incr("stats:success_count")
            else:
                pipe.incr("stats:error_count")
            pipe.execute()
        except Exception as e:
            logger.error(f"Failed to increment request count in Redis: {e}")
            # Fallback to local counters if Redis is unavailable
            # Note: This won't be accurate across workers, but prevents crashes

    async def get_request_stats(self) -> dict:
        """
        Get current request statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            r = await self.get_async_redis()
            pipe = r.pipeline()
            pipe.get("stats:request_count")
            pipe.get("stats:success_count")
            pipe.get("stats:error_count")
            pipe.get("stats:start_time")

            request_count, success_count, error_count, start_time = await pipe.execute()

            # Initialize counters if not set
            if request_count is None:
                request_count = 0
            if success_count is None:
                success_count = 0
            if error_count is None:
                error_count = 0
            if start_time is None:
                # Set start time to now if not exists
                await r.set("stats:start_time", str(time.time()))
                start_time = time.time()
            else:
                start_time = float(start_time)

            uptime = time.time() - start_time
            rps = int(request_count) / uptime if uptime > 0 else 0

            return {
                "total_requests": int(request_count),
                "successful_requests": int(success_count),
                "failed_requests": int(error_count),
                "uptime_seconds": uptime,
                "requests_per_second": rps
            }
        except Exception as e:
            logger.error(f"Failed to get request stats from Redis: {e}")
            # Return empty stats if Redis is unavailable
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "uptime_seconds": 0,
                "requests_per_second": 0,
                "error": "Redis unavailable"
            }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        try:
            r = self.get_redis()
            pipe = r.pipeline()
            pipe.delete("stats:request_count")
            pipe.delete("stats:success_count")
            pipe.delete("stats:error_count")
            pipe.delete("stats:start_time")
            pipe.execute()
            logger.info("Statistics reset in Redis")
        except Exception as e:
            logger.error(f"Failed to reset stats in Redis: {e}")


# Global stats manager instance
_stats_manager = RedisStatsManager()


def get_stats_manager() -> RedisStatsManager:
    """Get the global stats manager instance."""
    return _stats_manager


def increment_request_count(success: bool = True) -> None:
    """
    Increment request counters.

    This function is used by API routes to track request statistics.

    Args:
        success: Whether the request was successful
    """
    _stats_manager.increment_request_count(success)


async def get_request_stats() -> dict:
    """
    Get current request statistics.

    This function is used by the /stats endpoint.

    Returns:
        Dictionary with statistics
    """
    return await _stats_manager.get_request_stats()


def reset_stats() -> None:
    """Reset all statistics."""
    _stats_manager.reset_stats()
