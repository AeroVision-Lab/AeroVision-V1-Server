#!/usr/bin/env python3
"""
Registration OCR 准确率测试脚本
测试 OCR 识别的准确率和置信度
"""

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import requests

# 注册号格式验证
REGISTRATION_PATTERN = r"^[A-Z]{1,2}-[A-HJ-NP-Z0-9]{1,5}$"


class RegistrationAccuracyTester:
    """注册号 OCR 准确率测试器"""

    def __init__(self, base_url: str, data_dir: str, output_file: str = None):
        self.base_url = base_url.rstrip('/')
        self.data_dir = Path(data_dir)
        self.output_file = output_file

        # 统计数据
        self.results = []
        self.latencies = []

    def parse_ground_truth(self, filename: str) -> str:
        """从文件名解析真实注册号"""
        # 假设文件名格式为: REGISTRATION-0001.jpg
        # 例如: B-1234-0001.jpg
        parts = filename.replace('.jpg', '').split('-')
        if parts:
            # 找到第一个匹配注册号格式的部分
            import re
            for part in parts:
                if re.match(REGISTRATION_PATTERN, part):
                    return part
            # 如果没有找到，返回第一个部分作为备用
            return parts[0]
        return filename

    def load_test_images(self) -> List[Tuple[Path, str]]:
        """加载测试图片和真实标签"""
        images = []
        for img_file in self.data_dir.glob('*.jpg'):
            ground_truth = self.parse_ground_truth(img_file.name)
            images.append((img_file, ground_truth))
        return images

    def test_single_image(self, image_path: Path, ground_truth: str) -> Dict:
        """测试单张图片"""
        # 加载图片
        with open(image_path, 'rb') as f:
            img_data = f.read()
        base64_img = base64.b64encode(img_data).decode()

        # 发送API请求
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/registration",
                json={'image': base64_img},
                timeout=120  # Qwen API 可能较慢
            )
            duration = (time.time() - start_time) * 1000  # 毫秒

            if response.status_code == 200:
                result = response.json()
                registration = result['registration']
                confidence = result['confidence']
                raw_text = result.get('raw_text', '')

                # 验证注册号格式
                import re
                is_valid = bool(re.match(REGISTRATION_PATTERN, registration))

                return {
                    'success': True,
                    'ground_truth': ground_truth,
                    'predicted': registration,
                    'is_match': registration == ground_truth,
                    'confidence': confidence,
                    'raw_text': raw_text,
                    'is_valid_format': is_valid,
                    'latency_ms': duration,
                    'image': image_path.name
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status}",
                    'ground_truth': ground_truth,
                    'latency_ms': duration,
                    'image': image_path.name
                }

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return {
                'success': False,
                'error': str(e),
                'ground_truth': ground_truth,
                'latency_ms': duration,
                'image': image_path.name
            }

    def run(self):
        """运行完整的测试"""
        print(f"🧪 开始注册号 OCR 准确率测试...")
        print(f"   基础URL: {self.base_url}")
        print(f"   数据目录: {self.data_dir}")
        print()

        # 加载测试图片
        test_images = self.load_test_images()
        print(f"   找到 {len(test_images)} 张测试图片")
        print()

        # 测试每张图片
        for i, (image_path, ground_truth) in enumerate(test_images, 1):
            print(f"   测试 {i}/{len(test_images)}: {image_path.name}")
            result = self.test_single_image(image_path, ground_truth)
            self.results.append(result)

            if result['success']:
                self.latencies.append(result['latency_ms'])

                # 显示结果
                status = "✓" if result['is_match'] else "✗"
                print(f"      {status} 真实: {ground_truth}, 预测: {result['predicted']}, 置信度: {result['confidence']:.2f}")
            else:
                print(f"      ✗ 错误: {result.get('error', 'Unknown')}")

        # 计算指标
        metrics = self.calculate_metrics()
        self.print_results(metrics)

        # 保存结果
        if self.output_file:
            self.save_results(metrics)

        return metrics

    def calculate_metrics(self) -> Dict:
        """计算评估指标"""
        successful_results = [r for r in self.results if r['success']]
        total_count = len(self.results)
        success_count = len(successful_results)

        if success_count == 0:
            return {
                'total_images': total_count,
                'successful_tests': success_count,
                'failed_tests': total_count - success_count,
                'accuracy': 0.0,
                'valid_format_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_latency_ms': 0.0,
                'p50_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0
            }

        # 准确率
        correct_predictions = sum(1 for r in successful_results if r['is_match'])
        accuracy = correct_predictions / success_count

        # 格式有效性
        valid_format = sum(1 for r in successful_results if r['is_valid_format'])
        valid_format_rate = valid_format / success_count

        # 平均置信度
        avg_confidence = sum(r['confidence'] for r in successful_results) / success_count

        # 延迟统计
        avg_latency = sum(self.latencies) / len(self.latencies)
        sorted_latencies = sorted(self.latencies)
        p50_latency = sorted_latencies[len(sorted_latencies) // 2]
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        # 置信度分布
        high_conf_count = sum(1 for r in successful_results if r['confidence'] >= 0.8)
        high_conf_rate = high_conf_count / success_count

        return {
            'total_images': total_count,
            'successful_tests': success_count,
            'failed_tests': total_count - success_count,
            'accuracy': accuracy,
            'valid_format_rate': valid_format_rate,
            'avg_confidence': avg_confidence,
            'high_confidence_rate': high_conf_rate,
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'min_latency_ms': min(self.latencies),
            'max_latency_ms': max(self.latencies),
            'throughput_rps': success_count / (sum(self.latencies) / 1000),
            'detailed_results': successful_results
        }

    def print_results(self, metrics: Dict):
        """打印测试结果"""
        print(f"{'='*60}")
        print(f"📊 注册号 OCR 准确率测试结果")
        print(f"{'='*60}")
        print(f"   总图片数: {metrics['total_images']}")
        print(f"   成功测试: {metrics['successful_tests']}")
        print(f"   失败测试: {metrics['failed_tests']}")
        print()
        print(f"   识别准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   格式有效性: {metrics['valid_format_rate']:.4f} ({metrics['valid_format_rate']*100:.2f}%)")
        print(f"   平均置信度: {metrics['avg_confidence']:.4f}")
        print(f"   高置信度比例: {metrics['high_confidence_rate']:.4f} ({metrics['high_confidence_rate']*100:.2f}%)")
        print()
        print(f"   平均延迟: {metrics['avg_latency_ms']:.2f}ms")
        print(f"   P50 延迟: {metrics['p50_latency_ms']:.2f}ms")
        print(f"   P95 延迟: {metrics['p95_latency_ms']:.2f}ms")
        print(f"   P99 延迟: {metrics['p99_latency_ms']:.2f}ms")
        print(f"   最小延迟: {metrics['min_latency_ms']:.2f}ms")
        print(f"   最大延迟: {metrics['max_latency_ms']:.2f}ms")
        print(f"   吞吐量: {metrics['throughput_rps']:.2f} RPS")
        print()

    def save_results(self, metrics: Dict):
        """保存结果到文件"""
        # 创建输出字典（去除详细结果以减少文件大小）
        output_metrics = metrics.copy()
        if 'detailed_results' in output_metrics:
            del output_metrics['detailed_results']

        with open(self.output_file, 'w') as f:
            json.dump({'metrics': output_metrics, 'results': self.results}, f, indent=2)

        print(f"   📄 结果已保存到: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='注册号 OCR 准确率测试脚本')
    parser.add_argument('--base-url', required=True, help='API基础URL')
    parser.add_argument('--data-dir', required=True, help='测试数据目录')
    parser.add_argument('--output', help='结果输出文件（JSON格式）')

    args = parser.parse_args()

    # 创建测试器
    tester = RegistrationAccuracyTester(
        base_url=args.base_url,
        data_dir=args.data_dir,
        output_file=args.output
    )

    # 运行测试
    metrics = tester.run()

    return metrics


if __name__ == '__main__':
    main()
