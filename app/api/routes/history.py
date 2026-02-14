"""
History API routes.
提供历史审核记录管理接口
"""

from typing import List

from fastapi import APIRouter, HTTPException

from app.core import get_logger
from app.services.history_service import get_history_service

router = APIRouter()
logger = get_logger("api.history")


@router.post("/push")
async def push_historical_records(records: List[dict]) -> dict:
    """
    Push historical audit records to vector database.

    将历史审核记录推送到向量数据库，用于后续的相似度检索和新类别发现。

    Args:
        records: List of historical record dictionaries containing:
            - id: Record ID (required)
            - image_url: Optional image URL
            - aircraft_type: Aircraft type (required)
            - airline: Airline (required)
            - aircraft_confidence: Aircraft type confidence (required)
            - airline_confidence: Airline confidence (required)
            - timestamp: Optional timestamp
            - metadata: Optional metadata with image_data (base64)

    Returns:
        Dictionary with push results:
            - success: Overall success status
            - total: Total number of records
            - added: Number of successfully added records
            - failed: Number of failed records

    Raises:
        HTTPException: If vector database is not available or push fails.
    """
    try:
        history_service = get_history_service()
        result = history_service.push_records(records)
        return result

    except Exception as e:
        logger.error(f"历史记录推送失败: {e}")
        status_code = 503 if "向量数据库功能未启用" in str(e) else 500
        raise HTTPException(
            status_code=status_code,
            detail=str(e)
        )


@router.get("/stats")
async def get_history_stats() -> dict:
    """
    Get vector database statistics.

    获取历史记录统计信息，包括总记录数、机型分布、航司分布等。

    Returns:
        Dictionary with statistics:
            - available: Whether vector database is available
            - total_records: Total number of records
            - aircraft_types: Distribution of aircraft types
            - airlines: Distribution of airlines
            - message: Error message if not available

    Raises:
        HTTPException: If statistics retrieval fails.
    """
    try:
        history_service = get_history_service()
        stats = history_service.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
