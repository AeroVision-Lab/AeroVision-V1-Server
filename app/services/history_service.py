"""
History service for managing historical audit records with vector database.
"""

import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

from app.core import get_logger, get_settings
from app.core.exceptions import AerovisionException
from app.inference.factory import InferenceFactory

logger = get_logger("history_service")


class HistoryService:
    """Service for managing historical audit records with vector database."""

    def __init__(self):
        """Initialize the history service."""
        self._enhanced_predictor: Optional[Any] = None
        self._lock = threading.Lock()

    def _get_model_predictor(self) -> Optional[Any]:
        """
        Get or create the enhanced predictor with vector database.

        Uses InferenceFactory to load models with configuration.

        Returns:
            Enhanced predictor instance or None if not available.

        Raises:
            AerovisionException: If model loading fails.
        """
        if self._enhanced_predictor is not None:
            return self._enhanced_predictor

        with self._lock:
            if self._enhanced_predictor is not None:
                return self._enhanced_predictor

            try:
                # Try to import aerovision_inference for vector database support
                try:
                    from aerovision_inference import ModelPredictor

                    settings = get_settings()
                    device = settings.device
                    model_dir = settings.model_dir

                    config = {
                        'aircraft': {
                            'path': str(Path(model_dir) / 'aircraft.pt'),
                            'device': device,
                            'image_size': 640
                        },
                        'airline': {
                            'path': str(Path(model_dir) / 'airline.pt'),
                            'device': device,
                            'image_size': 640
                        }
                    }

                    predictor = ModelPredictor(config)
                    self._enhanced_predictor = predictor
                    logger.info("模型预测器初始化完成")

                    return self._enhanced_predictor

                except ImportError as e:
                    logger.warning(f"aerovision_inference package not available: {e}")
                    logger.info("向量数据库功能未启用")
                    return None

            except Exception as e:
                logger.error(f"初始化模型预测器失败: {e}")
                raise AerovisionException(
                    code="MODEL_LOAD_ERROR",
                    message=f"无法初始化特征提取模型: {str(e)}"
                )

    def push_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Push historical audit records to vector database.

        Args:
            records: List of historical record dictionaries containing:
                - id: Record ID
                - image_url: Optional image URL
                - aircraft_type: Aircraft type
                - airline: Airline
                - aircraft_confidence: Aircraft type confidence
                - airline_confidence: Airline confidence
                - timestamp: Optional timestamp
                - metadata: Optional metadata with image_data (base64)

        Returns:
            Dictionary with push results:
                - success: Overall success status
                - total: Total number of records
                - added: Number of successfully added records
                - failed: Number of failed records
        """
        logger.info(f"收到历史记录推送请求，记录数: {len(records)}")

        # Check if vector database is available
        enhanced_predictor = self._get_model_predictor()
        if enhanced_predictor is None or not hasattr(enhanced_predictor, 'vector_db'):
            raise AerovisionException(
                code="VECTOR_DB_NOT_AVAILABLE",
                message="向量数据库功能未启用"
            )

        try:
            from aerovision_inference import VectorRecord
            from app.services.base import BaseService

            added_count = 0
            failed_count = 0

            for record in records:
                try:
                    # Load image: prioritize image_data (base64), then image_url
                    image_input = None

                    # Check for base64 image data in metadata or use image_url
                    if record.get('metadata') and 'image_data' in record['metadata']:
                        image_input = record['metadata']['image_data']
                    elif record.get('image_url'):
                        image_input = record['image_url']
                    else:
                        logger.warning(f"记录 {record.get('id')} 缺少图片数据，跳过此记录")
                        failed_count += 1
                        continue

                    # Load image using BaseService
                    try:
                        image = BaseService.load_image(image_input)
                    except Exception as load_error:
                        logger.error(f"加载图片失败 {record.get('id')}: {load_error}")
                        failed_count += 1
                        continue

                    # Get predictor
                    predictor = enhanced_predictor.predictor
                    if predictor is None:
                        logger.error(f"预测器未初始化，无法处理记录 {record.get('id')}")
                        failed_count += 1
                        continue

                    # Use YOLO embed method to extract feature vectors
                    try:
                        img_array = np.array(image)

                        aircraft_emb_tensor = predictor.aircraft_model.embed(
                            img_array,
                            imgsz=640,
                            device=predictor.device,
                            verbose=False
                        )[0]

                        airline_emb_tensor = predictor.airline_model.embed(
                            img_array,
                            imgsz=640,
                            device=predictor.device,
                            verbose=False
                        )[0]

                        # Convert to numpy arrays and flatten
                        if hasattr(aircraft_emb_tensor, 'cpu'):
                            aircraft_emb = aircraft_emb_tensor.cpu().numpy()
                        else:
                            aircraft_emb = np.array(aircraft_emb_tensor)

                        if hasattr(airline_emb_tensor, 'cpu'):
                            airline_emb = airline_emb_tensor.cpu().numpy()
                        else:
                            airline_emb = np.array(airline_emb_tensor)

                        aircraft_emb = aircraft_emb.flatten()
                        airline_emb = airline_emb.flatten()

                    except Exception as embed_error:
                        logger.error(f"特征提取失败 {record.get('id')}: {embed_error}")
                        failed_count += 1
                        continue

                    # Create VectorRecord
                    vector_record = VectorRecord(
                        id=record.get('id'),
                        image_path=record.get('image_url') or f"metadata_{record.get('id')}",
                        aircraft_embedding=aircraft_emb,
                        airline_embedding=airline_emb,
                        aircraft_type=record.get('aircraft_type', ''),
                        airline=record.get('airline', ''),
                        aircraft_confidence=record.get('aircraft_confidence', 0.0),
                        airline_confidence=record.get('airline_confidence', 0.0),
                        timestamp=record.get('timestamp', ''),
                        metadata=record.get('metadata')
                    )

                    if enhanced_predictor.vector_db.add_record(vector_record):
                        added_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    logger.error(f"添加记录 {record.get('id')} 失败: {e}")
                    failed_count += 1

            logger.info(f"历史记录推送完成: 成功={added_count}, 失败={failed_count}")

            return {
                "success": True,
                "total": len(records),
                "added": added_count,
                "failed": failed_count
            }

        except AerovisionException:
            raise
        except Exception as e:
            logger.error(f"历史记录推送失败: {e}")
            raise AerovisionException(
                code="INTERNAL_ERROR",
                message=f"历史记录推送失败: {str(e)}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get vector database statistics.

        Returns:
            Dictionary with statistics:
                - available: Whether vector database is available
                - total_records: Total number of records
                - aircraft_types: Distribution of aircraft types
                - airlines: Distribution of airlines
                - message: Error message if not available
        """
        try:
            enhanced_predictor = self._get_model_predictor()
            if enhanced_predictor is None or not hasattr(enhanced_predictor, 'vector_db'):
                return {
                    "available": False,
                    "message": "向量数据库功能未启用"
                }

            stats = enhanced_predictor.get_db_statistics()
            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise AerovisionException(
                code="INTERNAL_ERROR",
                message=f"获取统计信息失败: {str(e)}"
            )


# Global service instance
_history_service: Optional[HistoryService] = None


def get_history_service() -> HistoryService:
    """
    Get or create the history service singleton.

    Returns:
        HistoryService instance.
    """
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service
