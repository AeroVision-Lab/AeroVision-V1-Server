"""
Unit tests for Redis client.

Tests Redis statistics manager with mocking to avoid requiring
an actual Redis server during tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytest_plugins = ('pytest_asyncio',)

from app.core.redis_client import (
    RedisStatsManager,
    get_stats_manager,
    increment_request_count,
    get_request_stats,
    reset_stats,
)


class TestRedisStatsManager:
    """Tests for RedisStatsManager."""

    def test_get_stats_manager_returns_singleton(self):
        """Test that get_stats_manager returns the same instance."""
        manager1 = get_stats_manager()
        manager2 = get_stats_manager()
        assert manager1 is manager2

    def test_increment_fallback_on_error(self):
        """Test that increment falls back gracefully on Redis error."""
        manager = RedisStatsManager()
        # Should not raise exception even if Redis is unavailable
        manager.increment_request_count(success=True)

    @pytest.mark.asyncio
    async def test_get_request_stats_fallback_on_error(self):
        """Test that get_request_stats falls back gracefully on Redis error."""
        manager = RedisStatsManager()
        # Should not raise exception even if Redis is unavailable
        stats = await manager.get_request_stats()
        # Should return default stats when Redis is unavailable
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    @patch('app.core.redis_client._stats_manager')
    def test_increment_request_count_calls_manager(self, mock_manager):
        """Test that increment_request_count calls the manager."""
        increment_request_count(success=True)
        mock_manager.increment_request_count.assert_called_once_with(True)

    @patch('app.core.redis_client._stats_manager')
    async def test_get_request_stats_calls_manager(self, mock_manager):
        """Test that get_request_stats calls the manager."""
        mock_manager.get_request_stats = AsyncMock(return_value={"total": 10})
        stats = await get_request_stats()
        mock_manager.get_request_stats.assert_called_once()
        assert stats == {"total": 10}

    @patch('app.core.redis_client._stats_manager')
    def test_reset_stats_calls_manager(self, mock_manager):
        """Test that reset_stats calls the manager."""
        reset_stats()
        mock_manager.reset_stats.assert_called_once()
