"""
请求模型定义
"""

from enum import Enum
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ReviewType(str, Enum):
    """审核类型枚举"""
    QUALITY = "quality"
    AIRCRAFT = "aircraft"
    REGISTRATION = "registration"
    OCCLUSION = "occlusion"
    VIOLATION = "violation"


class HistoricalRecordRequest(BaseModel):
    """
    历史审核记录推送请求

    注意：出于安全考虑，不再支持从本地文件路径加载图片。
    请使用 image_url 或在 metadata 中提供 image_data (base64 编码)。
    """
    id: str = Field(description="记录 ID")
    image_path: Optional[str] = Field(
        default=None,
        description="图片路径（已弃用，仅用于记录标识，不用于加载图片）"
    )
    image_url: Optional[str] = Field(default=None, description="图片 URL（推荐）")
    aircraft_type: str = Field(description="机型")
    airline: str = Field(description="航司")
    aircraft_confidence: float = Field(description="机型置信度", ge=0, le=1)
    airline_confidence: float = Field(description="航司置信度", ge=0, le=1)
    registration: Optional[str] = Field(default=None, description="注册号")
    timestamp: Optional[datetime] = Field(default=None, description="时间戳")
    metadata: Optional[dict] = Field(
        default=None,
        description="额外元数据。可以通过 metadata['image_data'] 提供 base64 编码的图片数据"
    )


class ReviewRequest(BaseModel):
    """图片审核请求"""

    image_url: Optional[str] = Field(
        default=None,
        description="图片 URL，与 image_base64 二选一"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 编码的图片数据，与 image_url 二选一"
    )
    review_types: List[ReviewType] = Field(
        default=[
            ReviewType.QUALITY,
            ReviewType.AIRCRAFT,
            ReviewType.REGISTRATION,
            ReviewType.OCCLUSION,
            ReviewType.VIOLATION,
        ],
        description="需要执行的审核类型列表"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="审核完成后的回调 URL"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="附加元数据，会在响应中原样返回"
    )

    @model_validator(mode="after")
    def validate_image_source(self) -> "ReviewRequest":
        """验证必须提供图片来源"""
        if not self.image_url and not self.image_base64:
            raise ValueError("必须提供 image_url 或 image_base64")
        if self.image_url and self.image_base64:
            raise ValueError("image_url 和 image_base64 只能二选一")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_url": "https://example.com/aircraft.jpg",
                    "review_types": ["quality", "aircraft", "registration"]
                },
                {
                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "review_types": ["quality"]
                }
            ]
        }
    }
