#!/usr/bin/env python3
"""
Qwen3-VL-Flash vs Qwen3-VL-Plus 对比分析
"""

import json

flash_file = '/home/wlx/AeroVision-V1-Server/deployment_tests/accuracy_results_qwen_100.json'
plus_file = '/home/wlx/AeroVision-V1-Server/deployment_tests/accuracy_results_qwen3_vl_plus_100.json'

with open(flash_file) as f:
    flash = json.load(f)

with open(plus_file) as f:
    plus = json.load(f)

print('='*80)
print('📊 Qwen3-VL-Flash vs Qwen3-VL-Plus 对比分析')
print('='*80)

print(f'\n{"指标":<25} {"Qwen3-VL-Flash":<25} {"Qwen3-VL-Plus":<25}')
print('-'*75)
print(f'{"准确率":<25} {flash["accuracy"]*100:>6.2f}% (80/100)         {plus["accuracy"]*100:>6.2f}% (92/100)')
print(f'{"平均延迟":<25} {flash["avg_latency_ms"]:>8.2f}ms                  {plus["avg_latency_ms"]:>8.2f}ms')
print(f'{"吞吐量":<25} {flash["throughput_rps"]:>8.2f} RPS                  {plus["throughput_rps"]:>8.2f} RPS')
print(f'{"平均置信度":<25} {flash["avg_confidence"]:.4f}                  {plus["avg_confidence"]:.4f}')
print(f'{"高置信度占比":<25} {flash["high_confidence_count"]/flash["successful_tests"]*100:>5.1f}%                   {plus["high_confidence_count"]/plus["successful_tests"]*100:>5.1f}%')
print(f'{"错误数量":<25} {len([r for r in flash["results"] if not r["is_correct"]]):>3} 个                        {len([r for r in plus["results"] if not r["is_correct"]]):>3} 个')

print(f'\n' + '='*80)
print('📈 准确率提升分析')
print('='*80)
print(f'''
   Qwen3-VL-Flash:  {flash["accuracy"]*100:.2f}% (80/100)
   Qwen3-VL-Plus:   {plus["accuracy"]*100:.2f}% (92/100)
   准确率提升:      {(plus["accuracy"]-flash["accuracy"])*100:+.2f}% ({plus["correct_count"]-flash["correct_count"]} 张)
   错误减少:        {len([r for r in flash["results"] if not r["is_correct"]]) - len([r for r in plus["results"] if not r["is_correct"]])} 张 ({(1-len([r for r in plus["results"] if not r["is_correct"]])/len([r for r in flash["results"] if not r["is_correct"]]))*100:.1f}%)
''')

print('='*80)
print('⚡ 性能对比')
print('='*80)
print(f'''
   Qwen3-VL-Flash:  {flash["avg_latency_ms"]:.0f}ms/请求, {flash["throughput_rps"]:.2f} RPS
   Qwen3-VL-Plus:   {plus["avg_latency_ms"]:.0f}ms/请求, {plus["throughput_rps"]:.2f} RPS
   延迟增加:        {(plus["avg_latency_ms"]/flash["avg_latency_ms"]-1)*100:+.1f}%
   吞吐量降低:      {(plus["throughput_rps"]/flash["throughput_rps"]-1)*100:+.1f}%
''')

print('='*80)
print('💰 费用估算 (基于 100 次调用)')
print('='*80)
print(f'''
   假设 Flash: ¥0.001/次, Plus: ¥0.003/次
   Qwen3-VL-Flash:  ¥0.100
   Qwen3-VL-Plus:   ¥0.300
   费用增加:        ¥0.200 (约3倍)
''')

print('='*80)
print('🎯 推荐选择')
print('='*80)
print(f'''
   ✅ 优先选择 Qwen3-VL-Plus 如果：
      - 需要最高准确率 (92% vs 80%)
      - 错误率要求低 (8% vs 20%)
      - 对延迟不敏感（~5.4s vs ~1.7s）
      - 预算充足（费用约 3 倍）
      - 需要高置信度结果（0.97 vs 0.89）

   ✅ 选择 Qwen3-VL-Flash 如果：
      - 需要快速响应（~1.7s vs ~5.4s）
      - 吞吐量要求高（0.59 RPS vs 0.18 RPS）
      - 预算有限
      - 对错误率容忍度较高

   📊 混合方案建议：
      - 高价值场景：使用 Plus（关键航班、VIP航班）
      - 普通场景：使用 Flash（批量处理、非关键航班）
      - 或结合 PaddleOCR 作为备份方案
      - 考虑根据置信度动态切换：高置信度用Flash，低置信度用Plus

   🔧 优化建议：
      1. 增加 max_tokens 参数解决 JSON 解析问题（Flash有6次失败）
      2. 优化 prompt 避免过长的 reasoning 响应
      3. 考虑批处理提高吞吐量
      4. 实现缓存机制避免重复调用
''')
