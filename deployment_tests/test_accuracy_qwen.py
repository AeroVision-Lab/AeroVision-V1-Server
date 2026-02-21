#!/usr/bin/env python3
"""
Qwen OCR 准确率测试
从 /home/wlx/Aerovision-V1/data/labeled/ 中取 100 张图片进行测试
真实注册号从 CSV 文件读取
"""

import os
import sys
import json
import time
import base64
import csv
from pathlib import Path
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Aerovision-V1-inference"))

from PIL import Image
import io

# 导入 DashScope OCR 客户端
from dashscope_client import DashScopeOCRClient, DashScopeError


def load_ground_truth(csv_file: str) -> dict:
    """
    从 CSV 文件加载真实注册号

    Args:
        csv_file: CSV 文件路径

    Returns:
        dict: 文件名到真实注册号的映射
    """
    ground_truth = {}
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f:  # 使用 utf-8-sig 处理 BOM
            reader = csv.DictReader(f)
            # 确定正确的 filename 键名（考虑 BOM）
            filename_key = 'filename'
            if reader.fieldnames and '\ufeff' in reader.fieldnames[0]:
                filename_key = reader.fieldnames[0]

            for row in reader:
                filename = row.get(filename_key, '')
                registration = row['registration']
                ground_truth[filename] = registration
    except Exception as e:
        print(f"⚠️  无法读取 CSV 文件: {e}")
        print(f"   将使用文件名解析作为备用方案")
        import traceback
        traceback.print_exc()
    return ground_truth


def run_accuracy_test(
    data_dir: str = "/home/wlx/Aerovision-V1/data/",
    csv_file: str = "/home/wlx/Aerovision-V1/data/labels.csv",
    max_images: int = 100,
    output_file: str = None
):
    """
    运行准确率测试

    Args:
        data_dir: 测试图片根目录
        csv_file: CSV 标签文件
        max_images: 最大测试图片数
        output_file: 结果输出文件
    """
    print("="*60)
    print("🧪 Qwen OCR 准确率测试")
    print("="*60)
    print(f"\n   数据目录: {data_dir}")
    print(f"   CSV 文件: {csv_file}")
    print(f"   测试图片数: {max_images}")
    print()

    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误：DASHSCOPE_API_KEY 环境变量未设置")
        return None

    # 加载真实标签
    print("📋 加载真实标签...")
    ground_truth = load_ground_truth(csv_file)
    print(f"   加载了 {len(ground_truth)} 条标签")
    print()

    # 初始化客户端
    try:
        print("📝 初始化 Qwen 客户端...")
        client = DashScopeOCRClient(
            model="qwen3.5-plus",  # 使用 qwen3.5-plus 模型
            timeout=120  # 增加超时时间到 120 秒
        )
        print("✅ 客户端初始化成功\n")
    except DashScopeError as e:
        print(f"❌ 初始化失败: {e}")
        return None

    # 获取测试图片
    labeled_dir = Path(data_dir) / "labeled"
    if not labeled_dir.exists():
        print(f"❌ 错误：找不到目录 {labeled_dir}")
        return None

    all_images = sorted(list(labeled_dir.glob("*.jpg")) + list(labeled_dir.glob("*.jpeg")))
    test_images = all_images[:max_images]

    print(f"📷 找到 {len(all_images)} 张图片")
    print(f"   测试 {len(test_images)} 张图片\n")

    # 测试结果
    results = []
    latencies = []
    errors = []
    correct = 0
    total = 0
    high_conf = 0
    medium_conf = 0
    low_conf = 0
    accuracy = 0.0
    avg_latency = 0.0
    p50_latency = 0.0
    p95_latency = 0.0
    avg_confidence = 0.0

    # 测试每张图片
    for i, image_path in enumerate(test_images, 1):
        # 从 CSV 获取真实注册号
        filename = image_path.name
        ground_truth_reg = ground_truth.get(filename, "")

        if not ground_truth_reg:
            print(f"[{i:3d}/{len(test_images)}] ⚠️  {filename}")
            print(f"       跳过：CSV 中没有找到真实标签")
            continue

        try:
            start_time = time.time()

            # 加载图片
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 调用 OCR
            result = client.recognize(image)

            latency = (time.time() - start_time) * 1000
            predicted = result["registration"]
            confidence = result["confidence"]

            # 判断是否正确
            is_correct = predicted == ground_truth_reg
            if is_correct:
                correct += 1

            total += 1
            latencies.append(latency)

            # 记录结果
            results.append({
                "image": filename,
                "ground_truth": ground_truth_reg,
                "predicted": predicted,
                "is_correct": is_correct,
                "confidence": confidence,
                "latency_ms": latency
            })

            # 显示进度
            status = "✓" if is_correct else "✗"
            print(f"[{i:3d}/{len(test_images)}] {status} {filename}")
            print(f"       真实: {ground_truth_reg}, 预测: {predicted}, 置信度: {confidence:.2f}, 延迟: {latency:.0f}ms")

        except DashScopeError as e:
            error_msg = str(e)
            errors.append({
                "image": filename,
                "ground_truth": ground_truth_reg,
                "error": error_msg
            })
            print(f"[{i:3d}/{len(test_images)}] ✗ {filename}")
            print(f"       错误: {error_msg}")

        except Exception as e:
            errors.append({
                "image": filename,
                "ground_truth": ground_truth_reg,
                "error": str(e)
            })
            print(f"[{i:3d}/{len(test_images)}] ✗ {filename}")
            print(f"       错误: {e}")

    # 计算统计指标
    if total > 0:
        accuracy = correct / total
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p50_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        avg_confidence = sum(r["confidence"] for r in results) / len(results)

        # 置信度分布
        high_conf = sum(1 for r in results if r["confidence"] >= 0.8)
        medium_conf = sum(1 for r in results if 0.5 <= r["confidence"] < 0.8)
        low_conf = sum(1 for r in results if r["confidence"] < 0.5)

    print("\n" + "="*60)
    print("📊 测试结果")
    print("="*60)

    if total > 0:
        print(f"\n总体统计：")
        print(f"   总图片数: {len(test_images)}")
        print(f"   成功识别: {total}")
        print(f"   失败识别: {len(errors)}")
        print(f"\n准确率指标：")
        print(f"   识别准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   正确识别数: {correct}/{total}")
        print(f"\n置信度统计：")
        print(f"   平均置信度: {avg_confidence:.4f}")
        print(f"   高置信度 (≥0.8): {high_conf} ({high_conf/total*100:.1f}%)")
        print(f"   中置信度 (0.5-0.8): {medium_conf} ({medium_conf/total*100:.1f}%)")
        print(f"   低置信度 (<0.5): {low_conf} ({low_conf/total*100:.1f}%)")
        print(f"\n延迟统计：")
        print(f"   平均延迟: {avg_latency:.2f}ms")
        print(f"   P50 延迟: {p50_latency:.2f}ms")
        print(f"   P95 延迟: {p95_latency:.2f}ms")
        print(f"   吞吐量: {total/(sum(latencies)/1000):.2f} RPS")

    if errors:
        print(f"\n错误列表 ({len(errors)}):")
        for err in errors[:10]:  # 只显示前10个
            print(f"   - {err['image']}: {err['error']}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors)-10} 个错误")

    # 错误案例分析
    if results:
        wrong_results = [r for r in results if not r["is_correct"]]
        if wrong_results:
            print(f"\n错误案例分析（前10个错误）：")
            for i, r in enumerate(wrong_results[:10], 1):
                print(f"   {i}. {r['image']}")
                print(f"      真实: {r['ground_truth']}, 预测: {r['predicted']}, 置信度: {r['confidence']:.2f}")

    # 保存结果
    if output_file:
        metrics = {
            "total_images": len(test_images),
            "successful_tests": total,
            "failed_tests": len(errors),
            "accuracy": accuracy if total > 0 else 0,
            "correct_count": correct,
            "avg_confidence": avg_confidence if results else 0,
            "high_confidence_count": high_conf,
            "medium_confidence_count": medium_conf,
            "low_confidence_count": low_conf,
            "avg_latency_ms": avg_latency if latencies else 0,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "throughput_rps": total/(sum(latencies)/1000) if latencies else 0,
            "results": results,
            "errors": errors
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 结果已保存到: {output_file}")

    return {
        "accuracy": accuracy if total > 0 else 0,
        "total": total,
        "correct": correct,
        "errors": len(errors)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Qwen OCR 准确率测试')
    parser.add_argument('--data-dir', default='/home/wlx/Aerovision-V1/data/', help='测试数据根目录')
    parser.add_argument('--csv-file', default='/home/wlx/Aerovision-V1/data/labels.csv', help='CSV 标签文件')
    parser.add_argument('--max-images', type=int, default=100, help='最大测试图片数')
    parser.add_argument('--output', default='accuracy_results_qwen.json', help='结果输出文件（JSON格式）')

    args = parser.parse_args()

    # 运行测试
    metrics = run_accuracy_test(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        max_images=args.max_images,
        output_file=args.output
    )

    if metrics and metrics["accuracy"] > 0:
        print(f"\n✅ 测试完成！准确率: {metrics['accuracy']*100:.2f}%")
    elif metrics:
        print(f"\n⚠️  测试完成，准确率: {metrics['accuracy']*100:.2f}%")

    return 0 if metrics else 1


if __name__ == "__main__":
    sys.exit(main())
