# Docker 部署指南

本文档说明如何使用 Docker 部署 AeroVision V1 Server。

## 前置要求

- Docker 20.10+
- Docker Compose 2.0+
- (可选) NVIDIA Docker Toolkit（用于 GPU 支持）

## 快速开始

### CPU 版本

```bash
# 构建镜像
docker build --build-arg DEVICE=cpu -t aerovision-server:cpu .

# 使用 Docker Compose 启动
docker compose -f docker-compose.cpu.yaml up -d

# 查看日志
docker logs -f aerovision-v1-server-cpu

# 停止服务
docker compose -f docker-compose.cpu.yaml down
```

### GPU 版本

```bash
# 确保 NVIDIA Docker Toolkit 已安装
# 构建镜像
docker build --build-arg DEVICE=gpu -t aerovision-server:gpu .

# 使用 Docker Compose 启动
docker compose up -d

# 查看日志
docker logs -f aerovision-v1-server

# 停止服务
docker compose down
```

## 配置说明

### 环境变量

创建 `.env` 文件（参考 `.env.docker.example`）：

```bash
# 服务配置
PORT=8001
WORKERS=1
DEBUG=false

# 模型配置
MODEL_DIR=./models
DEVICE=cuda
PRELOAD_MODELS=true

# OCR 配置
OCR_MODE=local
OCR_LANG=ch
USE_ANGLE_CLS=true

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=json

# 资源限制
CPU_LIMIT=4.0
MEMORY_LIMIT=8G
CPU_THREADS=4
```

### 卷挂载

- `./models:/app/models:ro` - 模型文件目录（只读）
- `./logs:/app/logs` - 日志文件目录

## 验证部署

### 健康检查

```bash
curl http://localhost:8001/api/v1/health
```

预期输出：
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "models_loaded": false,
  "gpu_available": true,
  "uptime_seconds": 10.0
}
```

### 服务状态

```bash
curl http://localhost:8001/api/v1/stats
```

### 容器状态

```bash
docker ps | grep aerovision-v1-server
```

## 故障排除

### 端口冲突

如果 8001 端口被占用，修改 `.env` 文件中的 `PORT` 变量：

```bash
PORT=8002
```

### GPU 不可用

如果是 CPU 版本但应用尝试使用 GPU，设置：

```bash
DEVICE=cpu
```

### 模型未加载

如果模型未正确加载，检查：

1. 模型文件是否正确挂载到 `/app/models`
2. 模型文件格式是否正确
3. 日志中的错误信息：

```bash
docker logs aerovision-v1-server-cpu
```

### 重新构建

如果代码或依赖有变化：

```bash
# 停止并删除容器
docker compose down

# 重新构建镜像（不使用缓存）
docker build --no-cache --build-arg DEVICE=cpu -t aerovision-server:cpu .

# 重新启动
docker compose -f docker-compose.cpu.yaml up -d
```

## 生产部署建议

1. **使用环境变量管理敏感信息**

2. **配置资源限制**

   在 `docker-compose.yaml` 中设置：
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4.0'
         memory: 8G
   ```

3. **日志管理**

   配置日志轮转：
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "100m"
       max-file: "3"
   ```

4. **健康检查**

   服务已配置健康检查，每 30 秒检查一次。

5. **自动重启**

   容器配置为 `restart: unless-stopped`。

## 多实例部署

如需部署多个实例：

```bash
# 复制 docker-compose.yaml
cp docker-compose.cpu.yaml docker-compose.cpu.yaml

# 修改服务名称和端口
# ports:
#   - "8002:8000"
# container_name: aerovision-v1-server-2

# 启动多个实例
docker compose -f docker-compose.cpu.yaml up -d
docker compose -f docker-compose.cpu-2.yaml up -d
```

## 性能优化

### GPU 版本

1. 使用合适的 CUDA 版本（默认 12.1）
2. 设置 `PRELOAD_MODELS=true` 启动时预加载模型
3. 适当调整 `WORKERS` 数量（通常为 GPU 数量）

### CPU 版本

1. 设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 等 CPU 线程数
2. 调整 `WORKERS` 数量（通常为 CPU 核心数）

## 监控

### 查看资源使用

```bash
docker stats aerovision-v1-server-cpu
```

### 查看日志

```bash
# 实时日志
docker logs -f aerovision-v1-server-cpu

# 最近的 100 行
docker logs --tail 100 aerovision-v1-server-cpu

# 查看特定时间范围
docker logs --since 2026-02-15T10:00:00 aerovision-v1-server-cpu
```

## 更新部署

```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker build --build-arg DEVICE=cpu -t aerovision-server:cpu .

# 重启容器
docker compose -f docker-compose.cpu.yaml up -d --force-recreate
```
