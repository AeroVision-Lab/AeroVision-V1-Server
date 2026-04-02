# AeroVision-V1-Server 代码审核报告

> 审核日期：2026-04-02
> 测试结果：85/85 通过，项目可正常跑通
> 审核范围：全量源码 + 测试覆盖率分析

---

## 严重问题（会导致运行时崩溃）

### [CRITICAL-1] `health.py` 硬依赖 `torch`，未安装时启动即崩溃

**文件：** `app/api/routes/health.py:6`

```python
import torch
...
gpu_available = torch.cuda.is_available()
```

`torch` 不在 `requirements/base.txt` 中，但无条件 import。只要没装 PyTorch，服务启动就会 `ImportError`，整个应用无法启动。

**修复方向：** 用 `try/except ImportError` 包裹，或改为检查 `INFERENCE_AVAILABLE`。

---

### [CRITICAL-2] 根目录 `core/` 与 `app/core/` 并存，两套模块冲突

**文件：** `core/logging.py`、`core/config.py` vs `app/core/logging.py`、`app/core/config.py`

根目录下存在旧版 `core/` 目录，`core/logging.py` 引用 `from .config import settings`，使用的是旧版 `Settings`（含 `app_version`、`cors_allow_credentials` 等字段）。`app/core/config.py` 是新版（含 `version`、`cors_methods` 等字段）。两套配置字段不一致，误引用旧版模块会导致 `AttributeError`。

**修复方向：** 删除根目录 `core/` 目录，统一使用 `app/core/`。

---

### [CRITICAL-3] 两套 `config.py` 字段命名不一致，存在混用风险

**文件：** `core/config.py` vs `app/core/config.py`

| 字段语义 | `core/config.py`（旧） | `app/core/config.py`（新） |
|---|---|---|
| 版本号 | `app_version` | `version` |
| CORS 凭证 | `cors_allow_credentials` | 无此字段 |
| CORS 方法 | `cors_allow_methods` | `cors_methods` |
| CORS 头 | `cors_allow_headers` | `cors_headers` |
| 最大图片大小 | `max_image_size`（bytes） | `max_image_size_mb`（MB） |

`app/main.py` 使用新版，`core/logging.py` 使用旧版，两套并存是历史遗留问题。

**修复方向：** 清理 `core/` 旧目录，统一字段命名。

---

## 中等问题（逻辑错误或设计缺陷）

### [MEDIUM-1] `include_quality=false` 时仍强制返回 quality 字段，语义错误

**文件：** `app/services/review_service.py:120-132`、`app/schemas/review.py:47-52`

`ReviewResult` schema 中 `quality` 和 `aircraft` 为必填字段（非 Optional）。当用户传 `include_quality=false&include_aircraft=false` 时，service 层会填入默认值（score=0.0, type_code="UNKNOWN"），导致响应中永远包含这两个字段，即使用户明确不需要。

```python
# review_service.py:128-132
if aircraft_result is None:
    aircraft_result = ReviewAircraftResult(
        type_code="UNKNOWN",
        confidence=0.0
    )
```

**修复方向：** 将 `ReviewResult.quality` 和 `ReviewResult.aircraft` 改为 `Optional`，或在 schema 层面区分"未请求"与"请求失败"两种状态。

---

### [MEDIUM-2] `clarity` 字段用 OCR confidence 代替，语义错误

**文件：** `app/services/review_service.py:115`、`app/services/review_service.py:285`

```python
clarity=reg_data.confidence  # Using OCR confidence as proxy
```

`clarity` 应表示注册号图像清晰度（图像质量维度），用 OCR 置信度代替语义不准确。该字段在 `ReviewRegistrationResult` 中为必填项，但实际传入的是错误含义的值。

**修复方向：** 从推理层获取真实清晰度分数，或将该字段改为 `Optional` 并在无法获取时返回 `null`。

---

### [MEDIUM-3] `asyncio.get_event_loop()` 在 Python 3.10+ 已废弃

**文件：**
- `app/services/quality_service.py:81`
- `app/services/registration_service.py:89`
- `app/services/_classifier_base.py:107, 159`
- `app/services/review_service.py:168, 187, 200`

```python
loop = asyncio.get_event_loop()  # 已废弃
```

Python 3.10+ 在异步上下文中调用 `get_event_loop()` 会触发 `DeprecationWarning`，Python 3.12 中行为已改变。当前项目运行在 Python 3.12.6。

**修复方向：** 全部替换为 `asyncio.get_running_loop()`。

---

### [MEDIUM-4] `history_service.py` 全局单例创建不是线程安全的

**文件：** `app/services/history_service.py:265-278`

```python
_history_service: Optional[HistoryService] = None

def get_history_service() -> HistoryService:
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()  # 竞态条件
    return _history_service
```

多 worker 场景下存在竞态条件。`HistoryService.__init__` 内部有 `threading.Lock`，但外层单例创建本身没有锁保护。

**修复方向：** 使用 `threading.Lock` 保护单例创建，或改用模块级初始化。

---

### [MEDIUM-5] `_classify_batch` 与 `_assess_batch`/`_recognize_batch` 异步模式不统一

**文件：** `app/services/_classifier_base.py:136`、`app/services/quality_service.py:111`、`app/services/registration_service.py:111`

`_classify_batch` 是 `async` 方法（内部用 `run_in_executor`），而 `_assess_batch` 和 `_recognize_batch` 是纯同步方法（在调用方被 `run_in_executor` 包裹）。两种模式不统一，增加维护复杂度，也使 `review_service` 的 batch 流程逻辑难以理解。

**修复方向：** 统一为同一种模式，推荐全部改为同步方法，由调用方统一用 `run_in_executor` 包裹。

---

### [MEDIUM-6] `quality.py` 路由中 `start_time` 声明后从未使用（死代码）

**文件：** `app/api/routes/quality.py:54`

```python
from datetime import datetime
...
start_time = datetime.utcnow()  # 声明了但从未使用
```

`datetime` import 和 `start_time` 赋值均为无效代码，batch 响应中也没有用到计时信息。

**修复方向：** 删除 `start_time = datetime.utcnow()` 及 `from datetime import datetime` import。

---

## 轻微问题（代码质量）

### [MINOR-1] `aircraft.py` 路由异常捕获顺序错误，`InferenceError` 永远不会被单独匹配

**文件：** `app/api/routes/aircraft.py:49`

```python
except (InferenceError, Exception) as e:
```

`Exception` 是 `InferenceError` 的父类，两者放在同一个 `except` 中，`InferenceError` 的存在毫无意义，等同于只写 `except Exception`。

**修复方向：** 拆分为两个独立的 `except` 块，或直接只保留 `except Exception`。

---

### [MINOR-2] `redis_client.py` 类型注解引用了未导入的 `Redis` 类型

**文件：** `app/core/redis_client.py:24`

```python
self._redis: Optional[Redis] = None  # Redis 未被导入
```

文件中只 `import redis`，没有 `from redis import Redis`，`Redis` 是未定义的名称。运行时不报错（仅注解），但静态类型检查（mypy/pyright）会报错。

**修复方向：** 添加 `from redis import Redis` 或改为 `Optional[redis.Redis]`。

---

### [MINOR-3] `pytest.ini` 未设置 `asyncio_default_fixture_loop_scope`，触发废弃警告

**文件：** `pytest.ini`

运行测试时出现：
```
PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
```

**修复方向：** 在 `pytest.ini` 中添加：
```ini
asyncio_default_fixture_loop_scope = function
```

---

## 汇总

| 级别 | 数量 | 说明 |
|---|---|---|
| CRITICAL | 3 | 启动崩溃风险 / 模块冲突 |
| MEDIUM | 6 | 逻辑错误 / 设计缺陷 |
| MINOR | 3 | 代码质量问题 |
| **合计** | **12** | |

**优先修复顺序：**
1. `CRITICAL-1`：`torch` 硬依赖（最高优先级，影响所有未装 PyTorch 的环境）
2. `CRITICAL-2/3`：清理旧 `core/` 目录
3. `MEDIUM-3`：替换废弃的 `asyncio.get_event_loop()`（Python 3.12 兼容性）
4. `MEDIUM-1`：修复 `include_quality=false` 时的语义错误
5. 其余问题按优先级排期处理
