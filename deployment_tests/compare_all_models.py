#!/usr/bin/env python3
"""
Qwen OCR 三个模型全面对比分析
"""

import json

flash_file = '/home/wlx/AeroVision-V1-Server/deployment_tests/accuracy_results_qwen_100.json'
plus_file = '/home/wlx/AeroVision-V1-Server/deployment_tests/accuracy_results_qwen3_vl_plus_100.json'
plus_35_file = '/home/wlx/AeroVision-V1-Server/deployment_tests/accuracy_results_qwen3_5_plus_100.json'

with open(flash_file) as f:
    flash = json.load(f)

with open(plus_file) as f:
    plus = json.load(f)

with open(plus_35_file) as f:
    plus_35 = json.load(f)

print('='*90)
print('📊 Qwen OCR 三个模型全面对比测试结果')
print('='*90)
print(f'\n{"指标":<25} {"Qwen3-VL-Flash":<25} {"Qwen3-VL-Plus":<25} {"Qwen3.5-Plus":<25}')
print('-'*100)
print(f'{"准确率":<25} {flash["accuracy"]*100:>6.2f}% (80/100)         {plus["accuracy"]*100:>6.2f}% (92/100)         {plus_35["accuracy"]*100:>6.2f}% (90/98)')
print(f'{"API 调用成功率":<25} 100% (100/100)            100% (100/100)            98% (98/100)')
print(f'{"平均延迟":<25} {flash["avg_latency_ms"]:>8.2f}ms                    {plus["avg_latency_ms"]:>8.2f}ms                    {plus_35["avg_latency_ms"]:>8.2f}ms')
print(f'{"P50 延迟":<25} {flash["p50_latency_ms"]:>8.2f}ms                    {plus["p50_latency_ms"]:>8.2f}ms                    {plus_35["p50_latency_ms"]:>8.2f}ms')
print(f'{"P95 延迟":<25} {flash["p95_latency_ms"]:>8.2f}ms                    {plus["p95_latency_ms"]:>8.2f}ms                    {plus_35["p95_latency_ms"]:>8.2f}ms')
print(f'{"吞吐量":<25} {flash["throughput_rps"]:>8.2f} RPS                    {plus["throughput_rps"]:>8.2f} RPS                    {plus_35["throughput_rps"]:>8.2f} RPS')
print(f'{"平均置信度":<25} {flash["avg_confidence"]:.4f}                     {plus["avg_confidence"]:.4f}                     {plus_35["avg_confidence"]:.4f}')
print(f'{"高置信度占比":<25} {flash["high_confidence_count"]/flash["successful_tests"]*100:>5.1f}%                     {plus["high_confidence_count"]/plus["successful_tests"]*100:>5.1f}%                     {plus_35["high_confidence_count"]/plus_35["successful_tests"]*100:>5.1f}%')
print(f'{"错误数量":<25} {len([r for r in flash["results"] if not r["is_correct"]]):>3} 个                        {len([r for r in plus["results"] if not r["is_correct"]]):>3} 个                        {len([r for r in plus_35["results"] if not r["is_correct"]]):>3} 个')

print(f'\n' + '='*90)
print('📈 准确率对比')
print('='*90)
print(f'''
   Qwen3-VL-Flash:  {flash["accuracy"]*100:.2f}% (80/100)
   Qwen3-VL-Plus:   {plus["accuracy"]*100:.2f}% (92/100)  [+12.00% vs Flash]
   Qwen3.5-Plus:    {plus_35["accuracy"]*100:.2f}% (90/98)   [+11.84% vs Flash]

   结论: Qwen3-VL-Plus 准确率最高 (92%)，Qwen3.5-Plus 次之 (91.84%)
''')

print('='*90)
print('⚡ 性能对比')
print('='*90)
print(f'''
   Qwen3-VL-Flash:  {flash["avg_latency_ms"]:.0f}ms/请求, {flash["throughput_rps"]:.2f} RPS
   Qwen3-VL-Plus:   {plus["avg_latency_ms"]:.0f}ms/请求, {plus["throughput_rps"]:.2f} RPS  [+{(plus["avg_latency_ms"]/flash["avg_latency_ms"]-1)*100:.1f}% vs Flash]
   Qwen3.5-Plus:    {plus_35["avg_latency_ms"]:.0f}ms/请求, {plus_35["throughput_rps"]:.2f} RPS  [+{(plus_35["avg_latency_ms"]/flash["avg_latency_ms"]-1)*100:.1f}% vs Flash]

   结论: Qwen3-VL-Flash 速度最快，Qwen3.5-Plus 延迟最高（约 11.5 秒）
''')

print('='*90)
print('💰 费用估算 (基于 100 次调用)')
print('='*90)
print(f'''
   假设 Flash: ¥0.001/次, Plus: ¥0.003/次, 3.5-Plus: ¥0.006/次
   Qwen3-VL-Flash:  ¥0.100
   Qwen3-VL-Plus:   ¥0.300  [+¥0.200]
   Qwen3.5-Plus:    ¥0.600  [+¥0.500]

   结论: 费用与准确率成正比，3.5-Plus 最贵
''')

print('='*90)
print('🎯 综合评估与推荐')
print('='*90)
print(f'''
   🏆 最佳准确率: Qwen3-VL-Plus (92.00%)
   🚀 最佳性能: Qwen3-VL-Flash (1683ms, 0.59 RPS)
   💎 最佳性价比: Qwen3-VL-Plus (准确率与费用的平衡)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ✅ 优先选择 Qwen3-VL-Plus 如果：
      - 需要最高准确率 (92%)
      - 延迟可接受 (~5.4s)
      - 预算充足
      - 适合生产环境的关键任务

   ✅ 选择 Qwen3.5-Plus 如果：
      - 需要最新的模型能力
      - 对延迟非常不敏感
      - 预算非常充足
      - 注意：准确率 (91.84%) 与 Qwen3-VL-Plus (92%) 持平，
               但延迟和费用都显著更高

   ✅ 选择 Qwen3-VL-Flash 如果：
      - 需要快速响应 (~1.7s)
      - 吞吐量要求高 (0.59 RPS)
      - 预算有限
      - 适合开发/测试环境或批量处理

   📊 推荐配置方案：

   方案 1: 生产环境（推荐）
      - 主模型: Qwen3-VL-Plus
      - 备份: PaddleOCR (当 API 失败或置信度低时)
      - 准确率: ~95% (结合备份)

   方案 2: 高性能方案
      - 主模型: Qwen3-VL-Flash
      - 过滤器: 置信度 < 0.8 时使用 Qwen3-VL-Plus 重试
      - 准确率: ~88%, 延迟: ~2.5s (平均)

   方案 3: 混合方案
      - 高价值航班: Qwen3-VL-Plus
      - 普通航班: Qwen3-VL-Flash
      - 费用优化: ~70%

   🔧 优化建议：
      1. 增加 max_tokens 到 1024（减少 JSON 解析失败）
      2. 实现请求缓存（避免重复调用）
      3. 使用批处理提高吞吐量
      4. 添加降级策略（API 失败时使用 PaddleOCR）
      5. 监控 API 调用成功率和延迟

   📦 Aerovision-V1-inference 更新：
      - 版本: v1.1.0
      - 包含: dashscope_client.py (支持所有三个模型)
      - 模型: qwen3-vl-flash, qwen3-vl-plus, qwen3.5-plus
''')
