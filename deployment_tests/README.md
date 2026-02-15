# AeroVision V1 Server - Deployment Tests

## 概述

这个目录包含AeroVision V1 Server的部署测试代码，用于验证Docker部署的功能正确性、性能和稳定性。

## 目录结构

```
deployment_tests/
├── README.md                          # 本文件
├── accuracy_test.py                   # 准确性测试
├── load_test.py                       # 负载测试
├── model_evaluation.py                # 模型评估
├── load_test.py (原tests目录)        # 单元负载测试
└── results/                           # 测试结果输出目录
    ├── cpu_accuracy_sample.json
    ├── cpu_comprehensive_report.json
    ├── cpu_load_test_short.json
    ├── final_test_report.json
    ├── TEST_SUMMARY.md
    └── load_test_results_*.json
```

## 测试类型

### 1. 准确性测试 (accuracy_test.py)

测试模型推理的准确性，验证识别结果的正确性。

```bash
cd deployment_tests
python accuracy_test.py
```

### 2. 负载测试 (load_test.py)

测试系统在高并发下的性能表现，包括响应时间和吞吐量。

```bash
cd deployment_tests
python load_test.py
```

### 3. 模型评估 (model_evaluation.py)

对模型进行全面的评估，包括不同场景下的性能指标。

```bash
cd deployment_tests
python model_evaluation.py
```

## 部署测试流程

### 1. 构建Docker镜像

```bash
# CPU 版本
docker build --build-arg DEVICE=cpu -t aerovision-server:cpu .

# GPU 版本
docker build --build-arg DEVICE=gpu -t aerovision-server:gpu .
```

### 2. 启动服务

```bash
# CPU 版本
docker-compose -f ../docker-compose.cpu.yaml up -d

# GPU 版本
docker-compose up -d
```

### 3. 运行测试

等待服务启动完成后（约1-2分钟），运行测试：

```bash
cd deployment_tests

# 运行所有测试
python accuracy_test.py --base-url http://localhost:8001 --data-dir /path/to/test/images
python load_test.py --cpu
python model_evaluation.py --cpu --data-dir /path/to/test/images

# 运行 GPU 版本测试
python accuracy_test.py --base-url http://localhost:8002 --data-dir /path/to/test/images
python load_test.py --gpu
python model_evaluation.py --gpu --data-dir /path/to/test/images

# 或者只运行特定测试
python load_test.py --url http://localhost:8001 --duration 60
```

**测试参数说明：**

- `--url`: 指定 API 基础 URL
- `--cpu`: 测试 CPU 版本（端口 8001）
- `--gpu`: 测试 GPU 版本（端口 8002）
- `--data-dir`: 指定测试图片目录
- `--sample-size`: 指定测试样本数量（model_evaluation）
- `--duration`: 指定每个并发级别的测试时长（load_test）

### 4. 查看结果

测试结果将保存在 `results/` 目录下：

- `accuracy_test.json` - 准确性测试结果
- `load_test_results_*.json` - 负载测试结果
- `evaluation_results.json` - 模型评估结果
- `TEST_SUMMARY.md` - 测试总结报告

## 测试配置

### 环境变量

创建 `.env` 文件配置测试参数：

```env
# 服务配置
SERVER_URL=http://localhost:8000
API_VERSION=v1

# 测试参数
CONCURRENT_USERS=10
TEST_DURATION=60
REQUEST_TIMEOUT=30
```

### 测试数据

将测试图片放置在 `test_data/` 目录下。

**重要说明：**
- 测试图片文件名应使用 ICAO 代码格式，例如：`A332-001.jpg`, `B77W-002.jpg`
- 测试脚本会自动将 ICAO 代码转换为模型输出的完整名称（如 `A330-200`, `777-300ER`）
- 内置了完整的 ICAO 代码映射表（参见 `icao_to_fullname_mapping.py`）

### API 服务端点

测试脚本默认使用以下 API 端点：
- CPU 版本：`http://localhost:8001`
- GPU 版本：`http://localhost:8002`
- 自定义：使用 `--url` 参数指定

## 注意事项

1. **测试结果不提交**: `results/` 目录下的JSON文件已在 `.gitignore` 中，不会被提交到版本控制。

2. **依赖安装**: 运行测试前，请确保安装了必要的依赖：
   ```bash
   pip install requests pytest locust
   ```

3. **服务健康检查**: 运行测试前，请确保服务正常运行：
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

4. **资源限制**: 负载测试可能会消耗大量系统资源，建议在非生产环境中运行。

5. **测试隔离**: 部署测试与单元测试不同，它们验证的是整个系统的部署和运行状态。

## 故障排查

### 服务无法启动

```bash
# 查看服务日志
docker logs aerovision-v1-server-cpu  # CPU 版本
docker logs aerovision-v1-server      # GPU 版本

# 检查端口占用
netstat -tuln | grep 8000
```

### 测试失败

```bash
# 检查服务是否正常运行
curl http://localhost:8001/api/v1/health  # CPU 版本
curl http://localhost:8002/api/v1/health  # GPU 版本

# 查看测试日志
python load_test.py --url http://localhost:8001 --duration 10
```

### 准确率异常低

如果测试结果显示准确率异常低（如 <10%），可能是以下原因：

1. **标签格式不匹配**：确保测试图片文件名使用 ICAO 代码格式
2. **数据路径错误**：检查 `--data-dir` 参数是否正确
3. **映射表缺失**：检查 `icao_to_fullname_mapping.py` 中是否包含需要的 ICAO 代码

解决方案：
```bash
# 查看测试脚本识别的图片数量
python model_evaluation.py --cpu --data-dir /path/to/images --sample-size 10

# 检查 ICAO 代码映射
python3 -c "from icao_to_fullname_mapping import ICAO_TO_FULLNAME; print(list(ICAO_TO_FULLNAME.items())[:10])"
```

## 持续集成

这些测试可以集成到CI/CD流程中：

```yaml
# 示例 CI 配置
- name: Run deployment tests
  run: |
    docker-compose up -d
    sleep 120
    cd deployment_tests
    python accuracy_test.py
    python load_test.py
    docker-compose down
```

## 参考资料

- [Docker部署文档](../DOCKER.md)
- [API文档](../README.md)
- [模型评估指南](../docs/model_evaluation.md)

## 最近更新

### 2026-02-15
- **修复准确率评估问题**：创建了 ICAO 代码到完整机型名称的映射表，解决了标签格式不匹配导致的准确率评估错误（从 8% 修正为预期 70-85%）
- **修复测试路径问题**：更新了所有测试脚本，移除了硬编码的路径，支持通过参数指定测试数据目录
- **改进测试脚本**：
  - `accuracy_test.py`：添加了 ICAO 代码映射支持
  - `model_evaluation.py`：重构标签加载逻辑，直接从文件名提取 ICAO 代码
  - `load_test.py`：支持自定义测试数据目录

### 已知问题
1. 需要 `/home/wlx/Aerovision-V1/data/labeled` 目录存在，或通过 `--data-dir` 参数指定
2. 首次运行需要下载 OCR 模型，启动时间较长（约 1-2 分钟）
