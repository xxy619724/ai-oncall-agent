# 异步任务系统设计文档（阶段一）

**日期**：2026-08-11
**目标**：将项目从"同步流式架构"升级为"最小可用异步任务系统"，实现请求-执行解耦

## 一、背景与问题

当前项目本质是"同步流式"架构：
- `graph.astream()` 在 HTTP 请求生命周期内同步执行
- SSE 连接断开则任务终止
- 无 TaskID 体系、无任务状态机、无取消/恢复能力
- 无 202 + Polling 两段式交互

对照企业级异步任务架构标准（触发层/执行层/交付层三层解耦），项目只在执行层做得相对完整，触发层和交付层都有明显短板。

## 二、设计目标

**阶段一最小目标**：实现请求-执行解耦，让任务脱离 HTTP 请求生命周期后台执行

**明确不做**（留到后续阶段）：
- Webhook / Cron 触发（阶段三）
- Artifact Store 分离（阶段二）
- 幂等性 IdempotencyKey（阶段二）
- 暂停/恢复接口（状态机已留扩展位，但不实现）
- Event Store 持久化（内存即可）
- 多 Worker 并发（阶段一固定单 Worker）

## 三、架构设计

### 3.1 状态机（6 种状态）

```
                    ┌─────────── cancelled
                    │
    created → queued → running → succeeded
                    │
                    └──→ failed
```

**合法流转**：
- `created → queued`（入队）
- `queued → running`（Worker 拉取）
- `queued → cancelled`（用户取消排队中任务）
- `running → succeeded`（执行成功）
- `running → failed`（执行异常）
- `running → cancelled`（用户取消执行中任务）

**终态**：`succeeded / failed / cancelled`（不可再流转）

### 3.2 数据模型（SQLite tasks 表）

| 字段 | 类型 | 说明 |
|------|------|------|
| task_id | TEXT PK | UUID4 |
| session_id | TEXT | 关联会话 |
| input_text | TEXT | 用户输入 |
| status | TEXT | 6种状态之一 |
| progress_completed | INTEGER | 已完成步骤数 |
| progress_total | INTEGER | 总步骤数 |
| result_text | TEXT | 最终响应（阶段一存全文） |
| error_message | TEXT | 失败原因 |
| created_at / updated_at | TEXT | ISO 时间戳 |
| started_at / ended_at | TEXT | 执行时间戳 |

### 3.3 API 设计（双轨并存）

**新增接口**（不破坏现有 `/api/aiops`）：
- `POST /api/tasks` → 202 + `{task_id}`
- `GET /api/tasks/{task_id}` → 任务状态详情
- `GET /api/tasks?limit=20` → 任务列表
- `GET /api/tasks/{task_id}/stream` → SSE 流式事件
- `POST /api/tasks/{task_id}/cancel` → 取消任务

### 3.4 代码结构

```
app/
├── models/
│   └── task.py              # Task 数据模型 + TaskStatus 枚举（~140行）
├── services/
│   ├── task_service.py      # TaskService + TaskStore（~330行）
│   └── task_worker.py       # TaskWorker 后台协程（~230行）
├── api/
│   └── task.py              # 任务 API 路由（~240行）
└── main.py                  # lifespan 启动 Worker（+10行）
```

### 3.5 数据流

**提交任务**：
```
POST /api/tasks
  → TaskService.submit() 写 TaskStore(status=created)
  → task_queue.put(task_id) → status=queued
  → 返回 202 + {task_id}
```

**Worker 执行**：
```
TaskWorker 后台循环:
  task_queue.get() → status=running
  → 调用 aiops_service.execute() 流式执行
  → 每个事件存入内存 event_store[task_id]
  → 检测取消信号 → status=cancelled
  → 完成: status=succeeded, result_text=response
  → 异常: status=failed, error_message=...
```

**查询/流式**：
```
GET /api/tasks/{id}        → TaskStore.get()
GET /api/tasks/{id}/stream → 从 event_store 读取 SSE 推送
POST /api/tasks/{id}/cancel → QUEUED 直接取消 / RUNNING 发送取消信号
```

## 四、关键设计点

### 4.1 任务恢复
服务重启时扫描 `status=running` 的任务，标记为 `failed`（error_message="服务重启中断"）。

### 4.2 事件存储
阶段一用内存 `dict[task_id, list[event]]`，不持久化。任务结束后保留事件供 SSE 端点读取。阶段二会替换为独立 EventStore 表。

### 4.3 任务超时
`config.task_timeout_seconds`（默认 300s），用 `asyncio.wait_for` 实现硬超时，超时标记 `failed`。

### 4.4 队列容量
`asyncio.Queue(maxsize=100)`，满时 `POST /api/tasks` 返回 503。

### 4.5 取消机制
- QUEUED 状态：直接流转到 CANCELLED
- RUNNING 状态：`asyncio.Event` 信号，Worker 在每个事件循环迭代检查

### 4.6 与现有代码的关系
- **不修改**：`aiops_service.py`、`executor.py`、`planner.py` 等核心 Agent 逻辑
- **新增**：TaskService 包装 `aiops_service.execute()`，把 AsyncGenerator 转为后台执行
- **保留**：现有 `/api/aiops` SSE 接口继续可用（双轨并存）

## 五、配置项

在 `config.py` 新增：
```python
task_db_path: str = "./data/tasks.db"         # 任务状态持久化 SQLite
task_queue_maxsize: int = 100                  # 任务队列容量
task_timeout_seconds: int = 300                # 任务执行硬超时
task_event_buffer_size: int = 200              # 单任务事件缓冲区大小
task_worker_concurrency: int = 1               # Worker 并发数（阶段一固定为 1）
```

## 六、验收标准

- [x] 状态机单元测试通过（5/5）
- [x] 所有模块可正常导入
- [ ] `POST /api/tasks` 返回 202 + task_id
- [ ] `GET /api/tasks/{task_id}` 返回任务状态
- [ ] `GET /api/tasks/{task_id}/stream` 推送 SSE 事件
- [ ] `POST /api/tasks/{task_id}/cancel` 取消任务
- [ ] 服务重启后 RUNNING 任务被标记为 FAILED
- [ ] 任务超时（>300s）被标记为 FAILED

## 七、后续演进路径

- **阶段二**：状态机扩展（pausing/paused/cancelling）+ 幂等性 + Artifact Store 分离
- **阶段三**：Webhook Receiver + Cron Scheduler + 版本控制
- **阶段四**：Event Store 持久化 + WebSocket Gateway + Callback Dispatcher
