#!/usr/bin/env python3
"""
模型效果测试脚本
评估准确率、召回率、F1、Top-1/Top-5、推理速度
"""

import argparse
import base64
import json
import time
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import requests

# Import ICAO code mapping
from icao_to_fullname_mapping import get_fullname


class AccuracyTester:
    """模型效果测试器"""

    def __init__(self, base_url: str, data_dir: str, output_file: str = None):
        self.base_url = base_url.rstrip('/')
        self.data_dir = Path(data_dir)
        self.output_file = output_file

        # 统计数据
        self.results = []
        self.latencies = []

    def parse_ground_truth(self, filename: str) -> str:
        """从文件名解析真实标签"""
        # 假设文件名格式为: LABEL-0001.jpg
        # 例如: A320-0001.jpg
        parts = filename.replace('.jpg', '').split('-')
        if parts:
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

        # 将 ICAO 代码转换为完整名称以匹配模型输出
        expected_fullname = get_fullname(ground_truth)

        # 发送API请求
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/aircraft",
                json={'image': base64_img},
                timeout=60
            )
            duration = (time.time() - start_time) * 1000  # 毫秒

            if response.status_code == 200:
                result = response.json()
                top1_pred = result['top1']['class']
                top1_conf = result['top1']['confidence']
                predictions = [p['class'] for p in result['predictions'][:5]]

                return {
                    'success': True,
                    'ground_truth': ground_truth,
                    'expected_fullname': expected_fullname,
                    'top1_prediction': top1_pred,
                    'top1_confidence': top1_conf,
                    'top5_predictions': predictions,
                    'top1_correct': top1_pred == expected_fullname,
                    'top5_correct': expected_fullname in predictions,
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
        print(f"🧪 开始模型效果测试...")
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
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'avg_latency_ms': 0.0,
                'p50_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0
            }

        # Top-1和Top-5准确率
        top1_correct = sum(1 for r in successful_results if r['top1_correct'])
        top5_correct = sum(1 for r in successful_results if r['top5_correct'])

        top1_accuracy = top1_correct / success_count
        top5_accuracy = top5_correct / success_count

        # 延迟统计
        avg_latency = statistics.mean(self.latencies)
        p50_latency = statistics.median(self.latencies)
        sorted_latencies = sorted(self.latencies)
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        # 按类别统计
        class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for result in successful_results:
            gt = result['ground_truth']
            class_stats[gt]['total'] += 1
            if result['top1_correct']:
                class_stats[gt]['correct'] += 1

        # 计算每个类别的准确率
        class_accuracies = {}
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                class_accuracies[class_name] = stats['correct'] / stats['total']

        return {
            'total_images': total_count,
            'successful_tests': success_count,
            'failed_tests': total_count - success_count,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'min_latency_ms': min(self.latencies),
            'max_latency_ms': max(self.latencies),
            'throughput_rps': success_count / (sum(self.latencies) / 1000),
            'class_accuracies': class_accuracies,
            'detailed_results': successful_results
        }

    def print_results(self, metrics: Dict):
        """打印测试结果"""
        print(f"{'='*60}")
        print(f"📊 模型效果测试结果")
        print(f"{'='*60}")
        print(f"   总图片数: {metrics['total_images']}")
        print(f"   成功测试: {metrics['successful_tests']}")
        print(f"   失败测试: {metrics['failed_tests']}")
        print()
        print(f"   Top-1 准确率: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
        print(f"   Top-5 准确率: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
        print()
        print(f"   平均延迟: {metrics['avg_latency_ms']:.2f}ms")
        print(f"   P50 延迟: {metrics['p50_latency_ms']:.2f}ms")
        print(f"   P95 延迟: {metrics['p95_latency_ms']:.2f}ms")
        print(f"   P99 延迟: {metrics['p99_latency_ms']:.2f}ms")
        print(f"   最小延迟: {metrics['min_latency_ms']:.2f}ms")
        print(f"   最大延迟: {metrics['max_latency_ms']:.2f}ms")
        print(f"   吞吐量: {metrics['throughput_rps']:.2f} RPS")
        print()

        # 打印各类别准确率
        if metrics.get('class_accuracies'):
            print(f"   各类别准确率:")
            sorted_classes = sorted(metrics['class_accuracies'].items(), key=lambda x: x[1], reverse=True)
            for class_name, accuracy in sorted_classes:
                print(f"      {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    def save_results(self, metrics: Dict):
        """保存结果到文件"""
        # 创建输出字典（去除详细结果以减少文件大小）
        output_metrics = metrics.copy()
        if 'detailed_results' in output_metrics:
            del output_metrics['detailed_results']

        with open(self.output_file, 'w') as f:
            json.dump({'metrics': output_metrics}, f, indent=2)

        print(f"   📄 结果已保存到: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='模型效果测试脚本')
    parser.add_argument('--base-url', required=True, help='API基础URL')
    parser.add_argument('--data-dir', required=True, help='测试数据目录')
    parser.add_argument('--output', help='结果输出文件（JSON格式）')

    args = parser.parse_args()

    # 创建测试器
    tester = AccuracyTester(
        base_url=args.base_url,
        data_dir=args.data_dir,
        output_file=args.output
    )

    # 运行测试
    metrics = tester.run()

    return metrics


if __name__ == '__main__':
    main()
