"""
Qwen VLM client with automatic failover support.

Tries each configured service in priority order (lowest priority value first).
Falls back to the next service on timeout or any HTTP error.
"""

import base64
import io
import json
import re
from typing import Any

import httpx
from PIL import Image

from app.core import get_logger, get_settings
from app.core.config import QwenServiceConfig

logger = get_logger("clients.qwen")


class QwenClientError(Exception):
    """Raised when all configured Qwen services have failed."""


class QwenClient:
    """
    Async Qwen VLM client with failover.

    On each call to ``generate()``, services are tried in ascending priority
    order.  The first successful response is returned together with the name
    of the service that handled it.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._services: list[QwenServiceConfig] = settings.qwen_services

        if not self._services:
            raise QwenClientError("No enabled Qwen services configured in QWEN_SERVICES_JSON")

        logger.info(
            "QwenClient initialised with %d service(s): %s",
            len(self._services),
            ", ".join(f"{s.name}(priority={s.priority})" for s in self._services),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[dict[str, Any], str]:
        """
        Send an image + prompt to Qwen and return the parsed JSON response.

        Args:
            image: PIL Image to analyse.
            prompt: Instruction prompt (should ask for a JSON response).

        Returns:
            ``(result_dict, service_name)`` — the parsed JSON payload and the
            name of the service that produced it.

        Raises:
            QwenClientError: If every configured service fails.
        """
        image_b64 = self._encode_image(image)
        errors: list[str] = []

        for service in self._services:
            try:
                result = await self._call_service(service, image_b64, prompt)
                logger.info("Qwen service '%s' succeeded", service.name)
                return result, service.name
            except Exception as exc:
                msg = f"{service.name}: {exc}"
                logger.warning("Qwen service failed — %s", msg)
                errors.append(msg)

        raise QwenClientError("All Qwen services failed: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Convert a PIL Image to a JPEG base64 string."""
        buf = io.BytesIO()
        # Ensure RGB (drop alpha channel if present)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    async def _call_service(
        service: QwenServiceConfig,
        image_b64: str,
        prompt: str,
    ) -> dict[str, Any]:
        """
        Execute a single API call against *service* and return the parsed JSON.

        Raises:
            httpx.HTTPStatusError: On non-2xx response.
            httpx.TimeoutException: On timeout.
            ValueError: If the response cannot be parsed as JSON.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if service.api_key:
            headers["Authorization"] = f"Bearer {service.api_key}"

        payload = {
            "model": service.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        async with httpx.AsyncClient(timeout=service.timeout) as client:
            response = await client.post(service.url, headers=headers, json=payload)
            response.raise_for_status()

        raw_content: str = response.json()["choices"][0]["message"]["content"]
        return QwenClient._parse_json_response(raw_content)

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any]:
        """
        Extract and parse a JSON object from the model's text response.

        Handles:
        - Plain JSON
        - JSON wrapped in ```json ... ``` code fences
        - JSON wrapped in generic ``` ... ``` fences
        """
        # Strip markdown code fences if present
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        json_str = fenced.group(1).strip() if fenced else content.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Cannot parse Qwen response as JSON: {exc}\nRaw content: {content[:300]}"
            ) from exc
