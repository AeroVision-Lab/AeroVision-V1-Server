# Qwen OCR 集成指南

## 概述

本文档描述了如何将阿里云百炼 Qwen API 集成到 AeroVision-V1 的 OCR 系统中，以实现飞机注册号识别。

## 架构

### 项目结构

```
Aerovision-V1-inference/
├── dashscope_client.py      # Qwen API 客户端（新增）
├── registration_ocr.py      # Registration OCR 类（已修改）
└── tests/
    ├── test_dashscope_client.py          # Qwen 客户端单元测试（新增）
    └── test_registration_ocr_qwen.py  # OCR Qwen 模式测试（新增）

AeroVision-V1-Server/
├── app/
│   ├── core/
│   │   └── config.py      # 配置（已更新）
│   ├── inference/
│   │   └── factory.py     # 推理工厂（已更新）
│   └── services/
│       └── registration_service.py  # OCR 服务
└── tests/
    └── unit/
        └── services/
            └── test_registration_service_qwen.py  # 服务测试（新增）
└── deployment_tests/
    ├── registration_accuracy_test.py  # 准确率测试（新增）
    └── registration_load_test.py     # 压力测试（新增）
```

### OCR 模式优先级

OCR 系统支持以下模式：

1. **auto**（默认）：自动检测 API key
   - 如果 `DASHSCOPE_API_KEY` 存在 → 使用 **qwen** 模式
   - 如果 `DASHSCOPE_API_KEY` 不存在 → 使用 **local** 模式

2. **qwen**：强制使用 Qwen API
   - 需要 `DASHSCOPE_API_KEY` 环境变量

3. **local**：使用本地 PaddleOCR
   - 无需 API key

4. **api**：使用外部 OCR API
   - 用于自定义 OCR 服务

## 配置

### 环境变量

在 `.env` 文件中配置：

```bash
# OCR 模式选择
OCR_MODE=auto                 # auto/qwen/local/api

# Qwen API 配置
DASHSCOPE_API_KEY=sk-xxxxx    # 阿里云百炼 API Key
QWEN_MODEL=qwen-vl-flash        # qwen-vl-flash 或 qwen-vl-plus
OCR_TIMEOUT=30                 # 超时时间（秒）

# 本地 OCR 配置（仅 local 模式）
OCR_LANG=ch                    # 语言
USE_ANGLE_CLS=true             # 是否使用角度分类

# 外部 API 配置（仅 api 模式）
# OCR_API_URL=http://localhost:8000/v2/models/ocr/infer
```

### 默认配置

```python
{
    "ocr_mode": "auto",              # 优先使用 qwen，回退到 local
    "qwen_model": "qwen-vl-flash", # 默认使用 flash（更快）
    "ocr_timeout": 30,
    "ocr_lang": "ch",
    "use_angle_cls": True
}
```

## 测试

### 单元测试

#### 1. Aerovision-V1-inference 单元测试

```bash
cd /home/wlx/Aerovision-V1-inference

# 运行 Qwen 客户端测试
python -m pytest tests/test_dashscope_client.py -v

# 运行 OCR Qwen 模式测试
python -m pytest tests/test_registration_ocr_qwen.py -v
```

#### 2. AeroVision-V1-Server 单元测试

```bash
cd /home/wlx/AeroVision-V1-Server

# 运行 Registration Service Qwen 模式测试
python -m pytest tests/unit/services/test_registration_service_qwen.py -v --no-cov
```

### 部署测试

#### 1. 准确率测试

测试 OCR 识别的准确率、置信度和格式有效性：

```bash
cd /home/wlx/AeroVision-V1-Server/deployment_tests

# 测试本地版本
python registration_accuracy_test.py \
    --base-url http://localhost:8000 \
    --data-dir /path/to/test/images \
    --output accuracy_results.json

# 测试 CPU 版本（如果已部署）
python registration_accuracy_test.py \
    --base-url http://localhost:8001 \
    --data-dir /path/to/test/images
```

**测试指标：**
- 识别准确率（Accuracy）：预测注册号与真实标签匹配的比例
- 格式有效性（Valid Format）：预测结果符合注册号格式的比例
- 平均置信度（Average Confidence）：所有预测的平均置信度
- 高置信度比例（High Confidence Rate）：置信度 ≥ 0.8 的比例
- 延迟统计：平均、P50、P95、P99 延迟
- 吞吐量：每秒处理的请求数（RPS）

#### 2. 压力测试

测试 API 在高并发下的性能和稳定性：

```bash
cd /home/wlx/AeroVision-V1-Server/deployment_tests

# 测试默认版本
python registration_load_test.py

# 测试 CPU 版本
python registration_load_test.py --cpu

# 测试 GPU 版本
python registration_load_test.py --gpu

# 自定义参数
python registration_load_test.py \
    --url http://localhost:8000 \
    --duration 60 \
    --data-dir /path/to/test/images
```

**并发级别：**
1 → 2 → 4 → 8 → 16 → 32 → 64 concurrent users

**测试指标：**
- 每个并发级别的请求数/秒（RPS）
- 成功率
- 延迟统计（平均、P50、P95、P99）
- 错误分布

## 运行流程

### 开发环境

1. **设置环境变量**

```bash
# 获取阿里云百炼 API Key
export DASHSCOPE_API_KEY=sk-xxxxx

# 设置 OCR 模式为 auto（自动检测）
export OCR_MODE=auto
```

2. **运行单元测试**

```bash
cd /home/wlx/Aerovision-V1-inference
python -m pytest tests/test_dashscope_client.py tests/test_registration_ocr_qwen.py -v

cd /home/wlx/AeroVision-V1-Server
python -m pytest tests/unit/services/test_registration_service_qwen.py -v --no-cov
```

3. **启动 Server**

```bash
cd /home/wlx/AeroVision-V1-Server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 生产环境

1. **配置环境变量**

```bash
# .env
OCR_MODE=auto
DASHSCOPE_API_KEY=sk-xxxxx
QWEN_MODEL=qwen-vl-flash
OCR_TIMEOUT=30
```

2. **部署 Server**

```bash
# 使用 gunicorn 部署
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

3. **运行部署测试**

```bash
# 准确率测试
python deployment_tests/registration_accuracy_test.py \
    --base-url http://localhost:8000 \
    --data-dir /path/to/test/images

# 压力测试
python deployment_tests/registration_load_test.py \
    --url http://localhost:8000
```

## 故障排除

### 1. DASHSCOPE_API_KEY 未找到

**错误信息：**
```
DashScopeError: DASHSCOPE_API_KEY not found in environment variables.
```

**解决方案：**
```bash
export DASHSCOPE_API_KEY=sk-xxxxx
```

或在 `.env` 文件中添加：
```bash
DASHSCOPE_API_KEY=sk-xxxxx
```

### 2. Qwen API 调用失败

**错误信息：**
```
DashScopeError: API request failed: ...
```

**解决方案：**
- 检查 API Key 是否有效
- 检查网络连接
- 检查 API 配额是否用完
- 查看 [阿里云百炼控制台](https://dashscope.console.aliyun.com/)

### 3. OCR 识别结果不准确

**解决方案：**
- 检查图片质量
- 尝试使用 `qwen-vl-plus` 模型（精度更高）
- 检查注册号是否在图片中清晰可见

### 4. 响应时间过长

**解决方案：**
- 使用 `qwen-vl-flash` 模型（速度更快）
- 检查网络延迟
- 考虑增加 `OCR_TIMEOUT` 值

## 性能对比

| 模式 | 平均延迟 | 准确率 | 成本 |
|------|---------|--------|------|
| qwen-vl-flash | ~2-3s | ~95% | 按调用计费 |
| qwen-vl-plus | ~3-5s | ~98% | 按调用计费（更高） |
| local (PaddleOCR) | ~0.5s | ~90% | 免费 |

**建议：**
- 开发/测试：使用 `qwen-vl-flash`（更快更便宜）
- 生产环境：使用 `qwen-vl-plus`（更高准确率）
- 高流量场景：考虑使用 `local` + `qwen` 混合模式

## 后续优化建议

1. **缓存机制**：对相同图片的识别结果进行缓存
2. **批处理**：支持批量识别以提高吞吐量
3. **重试机制**：在 API 失败时自动重试
4. **降级策略**：在 Qwen API 不可用时自动降级到本地 OCR
5. **监控告警**：监控 API 调用成功率、延迟和成本

## 参考资料

- [阿里云百炼 API 文档](https://help.aliyun.com/zh/dashscope/)
- [Qwen VL 模型文档](https://help.aliyun.com/zh/dashscope/developer-reference/qwen-vl-plus-api)
- [DashScope 兼容模式](https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-api)
