"""
历史记录 API 路由
提供历史审核记录推送接口
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends

from app.core import settings, get_logger
from app.schemas.request import HistoricalRecordRequest
from app.schemas.response import SimilarRecord
from app.services import get_review_service, ReviewService

router = APIRouter()
logger = get_logger("api.history")


@router.post("/push")
async def push_historical_records(
    records: List[HistoricalRecordRequest],
    review_service: ReviewService = Depends(get_review_service)
) -> dict:
    """
    推送历史审核记录

    将历史审核记录推送到向量数据库，用于后续的相似度检索和新类别发现。

    **请求参数**:
    - `records`: 历史记录列表

    **返回**:
    - 推送结果统计
    """
    logger.info(f"收到历史记录推送请求，记录数: {len(records)}")

    try:
        # 检查推理模块是否支持向量数据库
        if not hasattr(review_service, '_enhanced_predictor') or review_service._enhanced_predictor is None:
            raise HTTPException(
                status_code=503,
                detail="向量数据库功能未启用"
            )

        # 转换为 VectorRecord 并添加到数据库
        from aerovision_inference import VectorRecord
        import numpy as np

        added_count = 0
        failed_count = 0

        for record in records:
            try:
                # 注意：这里我们没有特征向量，需要先从图片提取
                # 实际实现中，应该使用已有的特征向量或者从图片重新提取
                # 这里简化处理，使用随机向量（实际应用中需要从模型提取）

                # TODO: 实际实现中需要从图片提取真实的特征向量
                aircraft_emb = np.random.rand(128)
                airline_emb = np.random.rand(128)

                vector_record = VectorRecord(
                    id=record.id,
                    image_path=record.image_path,
                    aircraft_embedding=aircraft_emb,
                    airline_embedding=airline_emb,
                    aircraft_type=record.aircraft_type,
                    airline=record.airline,
                    aircraft_confidence=record.aircraft_confidence,
                    airline_confidence=record.airline_confidence,
                    timestamp=record.timestamp.isoformat() if record.timestamp else None,
                    metadata=record.metadata
                )

                if review_service._enhanced_predictor.vector_db.add_record(vector_record):
                    added_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"添加记录 {record.id} 失败: {e}")
                failed_count += 1

        logger.info(f"历史记录推送完成: 成功={added_count}, 失败={failed_count}")

        return {
            "success": True,
            "total": len(records),
            "added": added_count,
            "failed": failed_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"历史记录推送失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"历史记录推送失败: {str(e)}"
        )


@router.get("/stats")
async def get_history_stats(
    review_service: ReviewService = Depends(get_review_service)
) -> dict:
    """
    获取历史记录统计信息

    **返回**:
    - 向量数据库统计信息
    """
    try:
        if not hasattr(review_service, '_enhanced_predictor') or review_service._enhanced_predictor is None:
            return {
                "available": False,
                "message": "向量数据库功能未启用"
            }

        stats = review_service._enhanced_predictor.get_db_statistics()
        return stats

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )
