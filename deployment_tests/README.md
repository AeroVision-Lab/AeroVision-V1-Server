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
python accuracy_test.py
python load_test.py
python model_evaluation.py

# 或者只运行特定测试
python load_test.py --concurrent 10 --duration 60
```

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
docker logs aerovision-v1-server

# 检查端口占用
netstat -tuln | grep 8000
```

### 测试失败

```bash
# 检查服务是否正常运行
curl http://localhost:8000/api/v1/health

# 查看测试日志
python load_test.py --verbose
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
