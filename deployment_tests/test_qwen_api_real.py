#!/usr/bin/env python3
"""
真实调用 Qwen API 的测试脚本
用于验证 API Key、余额和功能是否正常
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Aerovision-V1-inference"))

from PIL import Image
import io
import base64
from dashscope_client import DashScopeError


def test_api_key():
    """检查 API Key 是否配置"""
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        print("❌ 错误：DASHSCOPE_API_KEY 环境变量未设置")
        print("\n请设置 API Key：")
        print("  export DASHSCOPE_API_KEY=sk-xxxxx")
        print("\n或在 .env 文件中添加：")
        print("  DASHSCOPE_API_KEY=sk-xxxxx")
        return False

    # 脱敏显示
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"
    print(f"✅ API Key 已配置: {masked_key}")
    return True


def test_qwen_client():
    """测试 Qwen 客户端初始化"""
    try:
        from dashscope_client import DashScopeOCRClient, DashScopeError

        print("\n📝 初始化 Qwen 客户端...")

        client = DashScopeOCRClient(
            model="qwen3-vl-flash",
            timeout=30
        )

        print("✅ Qwen 客户端初始化成功")
        print(f"   模型: {client.model}")
        print(f"   API Base: {client.api_base}")

        return client
    except DashScopeError as e:
        if "DASHSCOPE_API_KEY" in str(e):
            print(f"❌ 错误：API Key 未配置或无效")
            print(f"   详情: {e}")
        else:
            print(f"❌ 错误：{e}")
        return None
    except Exception as e:
        print(f"❌ 意外错误：{e}")
        return None


def create_test_image():
    """创建测试图片"""
    print("\n🖼️  创建测试图片...")

    # 创建一张带有文字的测试图片
    from PIL import Image, ImageDraw, ImageFont

    # 创建白色背景
    image = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(image)

    # 尝试加载字体，如果失败使用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()

    # 绘制注册号文字
    text = "B-1234"
    # 计算文本位置居中
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (400 - text_width) // 2
    y = (200 - text_height) // 2

    draw.text((x, y), text, fill='black', font=font)

    print("✅ 测试图片创建成功")
    print(f"   内容: {text}")

    return image


def test_ocr_recognition(client):
    """测试 OCR 识别"""
    print("\n🔍 测试 OCR 识别...")

    try:
        # 创建测试图片
        test_image = create_test_image()

        # 调用识别
        print("   正在调用 Qwen API...")
        print("   这将在阿里云控制面板产生请求记录...")

        result = client.recognize(test_image)

        print("\n✅ OCR 识别成功！")
        print("\n   识别结果：")
        print(f"   - 注册号: {result['registration']}")
        print(f"   - 置信度: {result['confidence']:.2f}")
        print(f"   - 原始文本: {result['raw_text']}")

        if result['registration'] == "B-1234":
            print("\n✅ 识别结果正确！")
        else:
            print(f"\n⚠️  识别结果不正确，期望 'B-1234'，得到 '{result['registration']}'")

        # 检查置信度
        if result['confidence'] >= 0.8:
            print("✅ 置信度良好 (≥0.8)")
        else:
            print(f"⚠️  置信度较低 ({result['confidence']:.2f})")

        return True

    except DashScopeError as e:
        error_msg = str(e)

        # 检查常见的错误
        if "insufficient" in error_msg.lower() or "balance" in error_msg.lower():
            print("\n❌ 错误：账户余额不足")
            print("   请在阿里云百炼控制台充值：")
            print("   https://dashscope.console.aliyun.com/")

        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            print("\n❌ 错误：配额不足或已达调用限制")
            print("   请检查您的 API 配额")

        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print("\n❌ 错误：API 认证失败")
            print("   请检查 API Key 是否正确")

        elif "network" in error_msg.lower():
            print("\n❌ 错误：网络连接失败")
            print("   请检查网络连接和代理设置")

        else:
            print(f"\n❌ API 调用失败: {e}")

        return False

    except Exception as e:
        print(f"\n❌ 意外错误：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_image(client):
    """使用真实图片进行测试（可选）"""
    print("\n📷 是否要使用真实图片测试？")
    print("   (这将产生额外的 API 调用和费用)")

    # 这里可以添加使用真实图片的测试代码
    # 由于需要用户提供图片路径，暂时跳过
    print("   跳过真实图片测试")


def print_summary():
    """打印测试总结"""
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    print("\n✅ 如果测试成功：")
    print("   - 您应该能在阿里云控制面板看到请求记录")
    print("   - 控制面板: https://dashscope.console.aliyun.com/")
    print("\n⚠️  重要提示：")
    print("   - 每次成功调用都会产生费用")
    print("   - qwen-vl-flash 模型每次调用约 ¥0.01-0.02")
    print("   - 请确保账户有足够的余额")
    print("\n📚 相关文档：")
    print("   - 阿里云百炼: https://help.aliyun.com/zh/dashscope/")
    print("   - 计费说明: https://help.aliyun.com/zh/dashscope/developer-reference/billing")


def main():
    print("="*60)
    print("🧪 Qwen API 真实调用测试")
    print("="*60)

    # 检查 API Key
    if not test_api_key():
        return False

    # 测试客户端
    client = test_qwen_client()
    if not client:
        return False

    # 测试 OCR 识别
    success = test_ocr_recognition(client)

    # 打印总结
    print_summary()

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
