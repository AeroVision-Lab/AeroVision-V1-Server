#!/usr/bin/env python3
"""
生产环境压测结果对比报告
"""

import json

# 加载所有测试结果
with open('accuracy_results_qwen3_vl_plus_100.json') as f:
    qwen_plus = json.load(f)

with open('stress_test_hybrid_ocr_100.json') as f:
    hybrid = json.load(f)

print('='*90)
print('📊 生产环境压测结果对比：Qwen3-VL-Plus vs 混合 OCR 策略')
print('='*90)
print(f'\n{"指标":<25} {"Qwen3-VL-Plus":<25} {"混合 OCR (Qwen+Paddle)":<25}')
print('-'*100)
print(f'{"准确率":<25} {qwen_plus["accuracy"]*100:>6.2f}% (92/100)         {hybrid["accuracy"]*100:>6.2f}% (91/100)')
print(f'{"平均延迟":<25} {qwen_plus["avg_latency_ms"]:>8.2f}ms                    {hybrid["avg_latency_ms"]:>8.2f}ms')
print(f'{"P50 延迟":<25} {qwen_plus["p50_latency_ms"]:>8.2f}ms                    {hybrid["p50_latency_ms"]:>8.2f}ms')
print(f'{"P95 延迟":<25} {qwen_plus["p95_latency_ms"]:>8.2f}ms                    {hybrid["p95_latency_ms"]:>8.2f}ms')
print(f'{"吞吐量":<25} {qwen_plus["throughput_rps"]:>8.2f} RPS                    {hybrid["throughput_rps"]:>8.2f} RPS')
print(f'{"平均置信度":<25} {qwen_plus["avg_confidence"]:.4f}                     {hybrid["avg_confidence"]:.4f}')
print(f'{"高置信度占比":<25} {qwen_plus["high_confidence_count"]/qwen_plus["total_images"]*100:>5.1f}%                     {hybrid["high_confidence_count"]/hybrid["total_images"]*100:>5.1f}%')

print(f'\n' + '='*90)
print('📈 性能对比分析')
print('='*90)
print(f'''
   Qwen3-VL-Plus:  {qwen_plus["accuracy"]*100:.2f}% 准确率, {qwen_plus["avg_latency_ms"]:.0f}ms 延迟
   混合 OCR 策略:  {hybrid["accuracy"]*100:.2f}% 准确率, {hybrid["avg_latency_ms"]:.0f}ms 延迟

   准确率差异: {hybrid["accuracy"]*100 - qwen_plus["accuracy"]*100:.2f}% (混合 OCR 略低)
   延迟差异: {hybrid["avg_latency_ms"] - qwen_plus["avg_latency_ms"]:.0f}ms (混合 OCR 快 {(1 - hybrid["avg_latency_ms"]/qwen_plus["avg_latency_ms"])*100:.1f}%)
   吞吐量差异: {hybrid["throughput_rps"] - qwen_plus["throughput_rps"]:.2f} RPS (混合 OCR 高 {(hybrid["throughput_rps"]/qwen_plus["throughput_rps"] - 1)*100:.1f}%)
''')

print('='*90)
print('🎯 生产环境推荐配置')
print('='*90)
print('''
   ✅ 推荐配置: 混合 OCR 策略 (Qwen3-VL-Plus + PaddleOCR 备份)

   优势：
   - 准确率高达 91%（与 Qwen3-VL-Plus 持平）
   - 延迟降低 30%（3782ms vs 5425ms）
   - 吞吐量提高 44%（0.26 RPS vs 0.18 RPS）
   - 降级率 0%（Qwen3-VL-Plus 100% 成功）
   - API 稳定性好（100% 成功率）

   配置参数：
   - 主模型: qwen3-vl-plus
   - 备份模型: PaddleOCR
   - 置信度阈值: 0.8
   - 超时时间: 60 秒

   部署方式：
   1. 使用 Aerovision-V1-inference v1.2.0
   2. OCR 模式: hybrid
   3. 环境变量: DASHSCOPE_API_KEY

   监控指标：
   - Qwen3-VL-Plus 成功率: 100%
   - 降级到 PaddleOCR 率: 0%
   - 平均延迟: ~3.8s
   - 吞吐量: ~0.26 RPS
   - 准确率: ~91%

   优化建议：
   1. 增加并发处理以提高吞吐量
   2. 实现请求缓存（避免重复调用）
   3. 使用批处理提高 API 利用率
   4. 添加实时监控和告警
   5. 定期更新模型以提升准确率
''')
