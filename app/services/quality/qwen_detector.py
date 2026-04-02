"""
Qwen VLM-based image quality detector.

Covers semantic rules that require visual understanding beyond classical CV:

    Rules1.2.2  Glass reflection / glare / backlit / top-light
    Rules1.4.1  Dust spots / sensor dirt
    Rules1.5.1  Heat distortion / mirage
    Rules1.6.1  Compression artifacts
    Rules1.6.2  Cropping issues
    Rules1.6.3  Composition problems
    Rules1.8.3  Obstruction (fence, window, etc.)
    Rules2.3.1  Compliance (livery, registration visible)
    Rules2.4.1  Bad motivation (inappropriate angle/distance)

Each rule is evaluated by the VLM and returned as a RuleViolation if detected.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.clients.qwen_client import QwenClient, QwenClientError
from app.core import get_logger
from app.schemas.quality import RuleViolation

logger = get_logger("services.quality.qwen")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QwenResult:
    """All outputs from the Qwen detector."""

    violations: list[RuleViolation] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    service_name: str = ""  # which Qwen service handled the request


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_QWEN_PROMPT = """你是一个专业的航空摄影质量评估专家。请仔细分析这张飞机照片，检查以下质量问题：

**Rules1.2.2 - 玻璃反光/眩光/逆光/顶光**
- 检查是否有玻璃反光、镜头眩光、逆光拍摄导致主体过暗、或顶光导致的强烈阴影
- 严重程度：critical（主体完全不可见）、major（严重影响观感）、minor（轻微影响）

**Rules1.4.1 - 灰尘斑点/传感器污点**
- 检查画面中是否有明显的灰尘斑点或传感器污点（通常在天空等纯色区域明显）
- 严重程度：major（多处明显污点）、minor（少量污点）

**Rules1.5.1 - 热浪扭曲/海市蜃楼效应**
- 检查是否因高温导致的空气扭曲变形（常见于跑道、停机坪）
- 严重程度：major（严重扭曲）、minor（轻微扭曲）

**Rules1.6.1 - 压缩伪影**
- 检查是否有明显的 JPEG 压缩伪影、色块、马赛克
- 严重程度：major（严重伪影）、minor（轻微伪影）

**Rules1.6.2 - 裁切问题**
- 检查飞机主体是否被裁切（机翼、尾翼、机头等关键部位缺失）
- 严重程度：critical（关键部位缺失）、major（部分裁切）

**Rules1.6.3 - 构图问题**
- 检查构图是否合理：飞机在画面中的位置、大小、角度是否恰当
- 飞机过小、过大、位置偏离、角度不佳都属于构图问题
- 严重程度：major（严重构图问题）、minor（轻微构图问题）

**Rules1.8.3 - 遮挡物**
- 检查是否有栅栏、窗户、其他飞机、建筑物等遮挡主体
- 严重程度：critical（主体大部分被遮挡）、major（明显遮挡）、minor（轻微遮挡）

**Rules2.3.1 - 合规性**
- 检查飞机涂装、注册号是否清晰可见
- 检查是否符合航空摄影的基本要求
- 严重程度：major（关键信息不可见）、minor（信息不够清晰）

**Rules2.4.1 - 拍摄动机不佳**
- 检查拍摄角度、距离是否合适，是否能展现飞机特征
- 角度过于极端、距离过远/过近、视角不佳都属于动机问题
- 严重程度：major（严重影响展示效果）、minor（轻微影响）

请以 JSON 格式返回检测结果，格式如下：
{
  "violations": [
    {
      "rule_id": "Rules1.2.2",
      "rule_name": "玻璃反光/眩光",
      "severity": "major",
      "confidence": 0.85,
      "description": "画面存在明显的玻璃反光，影响飞机主体观感"
    }
  ],
  "suggestions": [
    "建议避免在玻璃窗内拍摄",
    "建议调整拍摄角度避免逆光"
  ]
}

注意：
1. 只返回确实存在的问题，不要返回不存在的问题
2. confidence 表示检测置信度，范围 0-1
3. 如果没有发现任何问题，violations 返回空数组
4. suggestions 是可选的改进建议，如果没有建议可以返回空数组
"""


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class QwenDetector:
    """
    Runs Qwen VLM-based semantic quality checks on a PIL Image and returns
    a ``QwenResult`` containing violations and suggestions.
    """

    def __init__(self) -> None:
        self._client = QwenClient()

    async def detect(self, image: Image.Image) -> QwenResult:
        """
        Run semantic quality checks via Qwen VLM.

        Args:
            image: PIL Image to analyse.

        Returns:
            QwenResult with violations and suggestions.

        Raises:
            QwenClientError: If all configured Qwen services fail.
        """
        try:
            response_dict, service_name = await self._client.generate(
                image=image,
                prompt=_QWEN_PROMPT,
            )
            logger.info("Qwen detection completed via service '%s'", service_name)

            result = QwenResult(service_name=service_name)
            result.violations = self._parse_violations(response_dict.get("violations", []))
            result.suggestions = response_dict.get("suggestions", [])

            logger.debug(
                "Qwen found %d violations, %d suggestions",
                len(result.violations),
                len(result.suggestions),
            )

            return result

        except QwenClientError as exc:
            logger.error("Qwen detection failed: %s", exc)
            raise

    # -----------------------------------------------------------------------
    # Parsing helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_violations(violations_data: list[dict[str, Any]]) -> list[RuleViolation]:
        """
        Parse raw violation dicts from Qwen response into RuleViolation objects.

        Args:
            violations_data: List of violation dicts from Qwen JSON response.

        Returns:
            List of validated RuleViolation objects.
        """
        result: list[RuleViolation] = []

        for item in violations_data:
            try:
                violation = RuleViolation(
                    rule_id=item["rule_id"],
                    rule_name=item["rule_name"],
                    severity=item["severity"],
                    confidence=float(item["confidence"]),
                    description=item["description"],
                    source="qwen",
                )
                result.append(violation)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid violation entry: %s — %s", item, exc)
                continue

        return result
