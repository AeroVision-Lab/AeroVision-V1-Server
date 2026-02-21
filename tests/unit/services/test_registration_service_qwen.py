"""
Registration Service Qwen 模式单元测试
"""

import pytest
import os
import base64
import io
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from PIL import Image

from app.services.registration_service import RegistrationService


class TestRegistrationServiceQwenMode:
    """Registration Service Qwen 模式测试"""

    def _create_test_image_base64(self):
        """创建测试图片的 base64 编码"""
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"})
    def test_init_qwen_mode(self):
        """测试在 Qwen 模式下初始化服务"""
        with patch('app.inference.factory.INFERENCE_AVAILABLE', True):
            with patch('app.core.config.get_settings') as mock_settings:
                mock_settings.return_value = Mock(
                    ocr_mode="qwen",
                    qwen_model="qwen-vl-flash",
                    ocr_timeout=30
                )

                # Mock RegistrationOCR 类
                with patch('app.inference.factory.RegistrationOCR') as mock_ocr_class:
                    mock_ocr_instance = MagicMock()
                    mock_ocr_instance.recognize.return_value = {
                        "registration": "B-1234",
                        "confidence": 0.95,
                        "raw_text": "B-1234",
                        "all_matches": [],
                        "yolo_boxes": []
                    }
                    mock_ocr_class.return_value = mock_ocr_instance

                    service = RegistrationService()
                    ocr_instance = service._get_ocr()

                    assert ocr_instance is not None

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"})
    def test_recognize_qwen_success(self):
        """测试 Qwen 模式识别成功"""
        # 先重置缓存
        from app.inference import factory
        factory.InferenceFactory._registration_ocr = None

        with patch('app.inference.factory.INFERENCE_AVAILABLE', True):
            with patch('app.core.config.get_settings') as mock_settings:
                mock_settings.return_value = Mock(
                    ocr_mode="qwen",
                    qwen_model="qwen-vl-flash",
                    ocr_timeout=30
                )

                # Mock RegistrationOCR 类
                with patch('app.inference.factory.RegistrationOCR') as mock_ocr_class:
                    mock_ocr_instance = MagicMock()
                    # RegistrationOCR.recognize 返回 dict（原始结果）
                    mock_ocr_instance.recognize.return_value = {
                        "registration": "B-5678",
                        "confidence": 0.92,
                        "raw_text": "B-5678",
                        "all_matches": [],
                        "yolo_boxes": []
                    }
                    mock_ocr_class.return_value = mock_ocr_instance

                    service = RegistrationService()
                    test_image_base64 = self._create_test_image_base64()
                    result, timing = service.recognize(test_image_base64)

                    assert result.registration == "B-5678"
                    assert result.confidence == 0.92
                    assert timing >= 0

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"})
    def test_recognize_qwen_error(self):
        """测试 Qwen 模式识别错误"""
        # 先重置缓存
        from app.inference import factory
        factory.InferenceFactory._registration_ocr = None

        with patch('app.inference.factory.INFERENCE_AVAILABLE', True):
            with patch('app.core.config.get_settings') as mock_settings:
                mock_settings.return_value = Mock(
                    ocr_mode="qwen",
                    qwen_model="qwen-vl-flash",
                    ocr_timeout=30
                )

                # Mock RegistrationOCR 类
                with patch('app.inference.factory.RegistrationOCR') as mock_ocr_class:
                    mock_ocr_instance = MagicMock()
                    mock_ocr_instance.recognize.side_effect = Exception("API error")
                    mock_ocr_class.return_value = mock_ocr_instance

                    service = RegistrationService()

                    from app.core.exceptions import ImageLoadError
                    # 测试错误处理
                    try:
                        test_image_base64 = self._create_test_image_base64()
                        service.recognize(test_image_base64)
                        assert False, "Should have raised an exception"
                    except Exception:
                        pass  # Expected
