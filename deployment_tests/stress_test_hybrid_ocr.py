#!/usr/bin/env python3
"""
混合 OCR 策略压测脚本

测试 Qwen3-VL-Plus + PaddleOCR 备份方案的性能和稳定性
"""

import os
import sys
import time
import json
import csv
import argparse
import logging
import statistics
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 动态导入 registration_ocr
aerovision_inference_path = Path(__file__).parent.parent.parent / 'Aerovision-V1-inference'
registration_ocr_path = aerovision_inference_path / 'registration_ocr.py'

spec = importlib.util.spec_from_file_location('registration_ocr', str(registration_ocr_path))
registration_ocr_module = importlib.util.module_from_spec(spec)
sys.modules['registration_ocr'] = registration_ocr_module
spec.loader.exec_module(registration_ocr_module)

RegistrationOCR = registration_ocr_module.RegistrationOCR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ground_truth(csv_file: str) -> Dict[str, str]:
    """
    从 CSV 文件加载真实注册号

    Args:
        csv_file: CSV 文件路径

    Returns:
        dict: 文件名到真实注册号的映射
    """
    ground_truth = {}
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            filename_key = 'filename'
            if reader.fieldnames and '\ufeff' in reader.fieldnames[0]:
                filename_key = reader.fieldnames[0]

            for row in reader:
                filename = row.get(filename_key, '')
                registration = row['registration']
                ground_truth[filename] = registration
    except Exception as e:
        logger.warning(f"无法读取 CSV 文件: {e}")
    return ground_truth


def stress_test_hybrid_ocr(
    data_dir: str,
    csv_file: str,
    max_images: int = 100,
    max_concurrent: int = 1,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    执行混合 OCR 压测

    Args:
        data_dir: 数据目录
        csv_file: CSV 标签文件
        max_images: 最大测试图片数
        max_concurrent: 最大并发数
        output_file: 输出文件路径

    Returns:
        压测结果
    """
    print('='*80)
    print('🧪 混合 OCR 策略压测')
    print('='*80)
    print(f'\n   数据目录: {data_dir}')
    print(f'   CSV 文件: {csv_file}')
    print(f'   测试图片数: {max_images}')
    print(f'   最大并发数: {max_concurrent}')

    # 加载真实标签
    print('\n📋 加载真实标签...')
    ground_truth = load_ground_truth(csv_file)
    print(f'   加载了 {len(ground_truth)} 条标签')

    # 初始化混合 OCR
    print('\n📝 初始化混合 OCR (Qwen3-VL-Plus + PaddleOCR)...')
    try:
        ocr = RegistrationOCR(
            mode='hybrid',
            qwen_model='qwen3-vl-plus',
            confidence_threshold=0.8,
            timeout=60
        )
        print('✅ 混合 OCR 初始化成功\n')
    except Exception as e:
        print(f'❌ 混合 OCR 初始化失败: {e}')
        return {
            'total_images': 0,
            'successful_tests': 0,
            'failed_tests': max_images,
            'accuracy': 0.0,
            'correct_count': 0,
            'avg_latency_ms': 0.0,
            'p50_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'throughput_rps': 0.0,
            'avg_confidence': 0.0,
            'high_confidence_count': 0,
            'medium_confidence_count': 0,
            'low_confidence_count': 0,
            'qwen_success_count': 0,
            'paddle_fallback_count': 0,
            'qwen_api_fail_count': 0,
            'results': [],
            'errors': []
        }

    # 收集测试图片
    data_path = Path(data_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = []
    for ext in image_extensions:
        all_images.extend(data_path.rglob(f'*{ext}'))

    if max_images:
        all_images = all_images[:max_images]

    print(f'📷 找到 {len(all_images)} 张图片')
    print(f'   测试 {len(all_images)} 张图片\n')

    # 测试结果
    results = []
    latencies = []
    errors = []

    # 统计
    correct = 0
    total = 0
    high_conf = 0
    medium_conf = 0
    low_conf = 0
    qwen_success = 0
    paddle_fallback = 0
    qwen_api_fail = 0

    # 执行压测
    start_time = time.time()

    for idx, image_path in enumerate(all_images, 1):
        filename = image_path.name
        ground_truth_reg = ground_truth.get(filename)

        if not ground_truth_reg:
            print(f'[  {idx:3}/{len(all_images)}] ⚠️  {filename}')
            print(f'       跳过：CSV 中没有找到真实标签')
            continue

        try:
            print(f'[  {idx:3}/{len(all_images)}] 🔄 {filename}', end='')

            # 执行 OCR 识别
            result_start = time.time()
            result = ocr.recognize(str(image_path))
            latency = (time.time() - result_start) * 1000  # 转换为毫秒

            predicted = result['registration']
            confidence = result['confidence']

            # 判断是否正确
            is_correct = (predicted.upper() == ground_truth_reg.upper())

            # 统计
            total += 1
            latencies.append(latency)

            if is_correct:
                correct += 1
                print(f'\r[  {idx:3}/{len(all_images)}] ✓ {filename}')
            else:
                print(f'\r[  {idx:3}/{len(all_images)}] ✗ {filename}')

            # 置信度统计
            if confidence >= 0.8:
                high_conf += 1
            elif confidence >= 0.5:
                medium_conf += 1
            else:
                low_conf += 1

            # 估算使用的是 Qwen 还是 PaddleOCR（基于置信度和结果）
            # 高置信度（>=0.8）且不是 PaddleOCR 的典型错误模式，认为是 Qwen
            # 低置信度（<0.8）或 PaddleOCR 的典型错误模式，认为是 PaddleOCR
            if confidence >= 0.8:
                qwen_success += 1
            else:
                paddle_fallback += 1

            print(f'       真实: {ground_truth_reg}, 预测: {predicted}, 置信度: {confidence:.2f}, 延迟: {latency:.0f}ms')

            # 保存结果
            results.append({
                'image': filename,
                'ground_truth': ground_truth_reg,
                'predicted': predicted,
                'confidence': confidence,
                'latency_ms': latency,
                'is_correct': is_correct
            })

        except Exception as e:
            total += 1
            errors.append({
                'image': filename,
                'error': str(e)
            })
            print(f'\r[  {idx:3}/{len(all_images)}] ❌ {filename}')
            print(f'       错误: {e}')

    total_time = time.time() - start_time

    # 计算统计指标
    accuracy = correct / total if total > 0 else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p50_latency = statistics.median(latencies) if latencies else 0.0
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (latencies[-1] if latencies else 0.0)
    throughput = total / total_time if total_time > 0 else 0.0
    avg_confidence = statistics.mean([r['confidence'] for r in results]) if results else 0.0

    # 输出结果
    print('\n' + '='*80)
    print('📊 压测结果')
    print('='*80)

    print(f'\n总体统计：')
    print(f'   总图片数: {total}')
    print(f'   成功识别: {total - len(errors)}')
    print(f'   失败识别: {len(errors)}')

    print(f'\n准确率指标：')
    print(f'   识别准确率: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'   正确识别数: {correct}/{total}')

    print(f'\n置信度统计：')
    print(f'   平均置信度: {avg_confidence:.4f}')
    print(f'   高置信度 (≥0.8): {high_conf} ({high_conf/total*100:.1f}%)')
    print(f'   中置信度 (0.5-0.8): {medium_conf} ({medium_conf/total*100:.1f}%)')
    print(f'   低置信度 (<0.5): {low_conf} ({low_conf/total*100:.1f}%)')

    print(f'\n延迟统计：')
    print(f'   平均延迟: {avg_latency:.2f}ms')
    print(f'   P50 延迟: {p50_latency:.2f}ms')
    print(f'   P95 延迟: {p95_latency:.2f}ms')
    print(f'   吞吐量: {throughput:.2f} RPS')

    print(f'\n混合 OCR 策略统计：')
    print(f'   Qwen3-VL-Plus 成功: {qwen_success} ({qwen_success/total*100:.1f}%)')
    print(f'   PaddleOCR 降级: {paddle_fallback} ({paddle_fallback/total*100:.1f}%)')
    print(f'   降级率: {paddle_fallback/total*100:.2f}%')

    # 错误列表
    if errors:
        print(f'\n错误列表 ({len(errors)}):')
        for error in errors:
            print(f'   - {error["image"]}: {error["error"]}')

    # 错误案例分析（前10个错误）
    error_results = [r for r in results if not r['is_correct']]
    if error_results:
        print(f'\n错误案例分析（前10个错误）:')
        for i, r in enumerate(error_results[:10], 1):
            print(f'   {i}. {r["image"]}')
            print(f'      真实: {r["ground_truth"]}, 预测: {r["predicted"]}, 置信度: {r["confidence"]:.2f}, 延迟: {r["latency_ms"]:.0f}ms')

    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_name': 'Hybrid OCR Stress Test (Qwen3-VL-Plus + PaddleOCR)',
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'data_dir': data_dir,
                    'csv_file': csv_file,
                    'max_images': max_images,
                    'max_concurrent': max_concurrent,
                    'ocr_mode': 'hybrid',
                    'qwen_model': 'qwen3-vl-plus',
                    'confidence_threshold': 0.8
                },
                'total_images': total,
                'successful_tests': total - len(errors),
                'failed_tests': len(errors),
                'accuracy': accuracy,
                'correct_count': correct,
                'avg_latency_ms': avg_latency,
                'p50_latency_ms': p50_latency,
                'p95_latency_ms': p95_latency,
                'throughput_rps': throughput,
                'avg_confidence': avg_confidence,
                'high_confidence_count': high_conf,
                'medium_confidence_count': medium_conf,
                'low_confidence_count': low_conf,
                'qwen_success_count': qwen_success,
                'paddle_fallback_count': paddle_fallback,
                'qwen_api_fail_count': qwen_api_fail,
                'results': results,
                'errors': errors
            }, f, indent=2, ensure_ascii=False)

        print(f'\n✅ 结果已保存到: {output_path}')

    return {
        'total_images': total,
        'successful_tests': total - len(errors),
        'failed_tests': len(errors),
        'accuracy': accuracy,
        'correct_count': correct,
        'avg_latency_ms': avg_latency,
        'p50_latency_ms': p50_latency,
        'p95_latency_ms': p95_latency,
        'throughput_rps': throughput,
        'avg_confidence': avg_confidence,
        'high_confidence_count': high_conf,
        'medium_confidence_count': medium_conf,
        'low_confidence_count': low_conf,
        'qwen_success_count': qwen_success,
        'paddle_fallback_count': paddle_fallback,
        'qwen_api_fail_count': qwen_api_fail,
        'results': results,
        'errors': errors
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='混合 OCR 策略压测')
    parser.add_argument('--data-dir', type=str,
                       default='/home/wlx/Aerovision-V1/data/',
                       help='数据目录')
    parser.add_argument('--csv-file', type=str,
                       default='/home/wlx/Aerovision-V1/data/labels.csv',
                       help='CSV 标签文件')
    parser.add_argument('--max-images', type=int, default=100,
                       help='最大测试图片数')
    parser.add_argument('--max-concurrent', type=int, default=1,
                       help='最大并发数')
    parser.add_argument('--output', type=str,
                       default='stress_test_hybrid_ocr_results.json',
                       help='输出文件路径')

    args = parser.parse_args()

    results = stress_test_hybrid_ocr(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        max_images=args.max_images,
        max_concurrent=args.max_concurrent,
        output_file=args.output
    )

    print(f'\n✅ 压测完成！准确率: {results["accuracy"]*100:.2f}%')
    print(f'   Qwen3-VL-Plus 成功: {results["qwen_success_count"]} ({results["qwen_success_count"]/results["total_images"]*100:.1f}%)')
    print(f'   PaddleOCR 降级: {results["paddle_fallback_count"]} ({results["paddle_fallback_count"]/results["total_images"]*100:.1f}%)')

    sys.exit(0 if results['accuracy'] >= 0.9 else 1)


if __name__ == '__main__':
    main()
