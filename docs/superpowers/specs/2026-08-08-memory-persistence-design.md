# 对话记忆持久化功能设计

**目标：** 将当前仅存于内存的对话记忆持久化到 SQLite + Redis 双层存储，服务重启后可恢复用户历史对话上下文，同时通过 Redis 摘要层加速上下文加载。

## 背景问题

### 当前记忆管理的现状

项目使用 LangGraph 的 `MemorySaver`（纯内存 checkpointer）管理对话历史：

```python
# rag_agent_service.py L110
from langgraph.checkpoint.memory import MemorySaver
self.checkpointer = MemorySaver()  # 纯内存，重启即丢失
```

**核心问题：** 服务一旦重启，所有用户的对话历史全部丢失，用户需要重新描述上下文。

### 现有机制（已实现，可复用）

| 机制 | 位置 | 说明 |
|---|---|---|
| `thread_id` 会话隔离 | `config_dict = {"configurable": {"thread_id": session_id}}` | 按 session_id 隔离对话 |
| `SummarizationMiddleware` | `rag_agent_service.py L147-154` | 上下文 70% 时自动压缩，生成摘要 SystemMessage |
| `trim_messages_middleware` | `rag_agent_service.py L42-79` | 保留最近 6 条消息（3 轮） |
| `get_session_history()` | `rag_agent_service.py L329-392` | 从 checkpointer 读取历史 |
| `clear_session()` | `rag_agent_service.py L394-413` | 从 checkpointer 删除线程 |

## 技术方案

### 调研结论：LangGraph 原生支持，无需自研

LangGraph 的 checkpointer 体系原生支持多种持久化后端：

| Checkpointer | 包名 | 存储类型 | 是否需额外部署 |
|---|---|---|---|
| `MemorySaver` | `langgraph`（内置） | 内存 | 否（当前使用） |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | SQLite 文件 | 否 |
| `PostgresSaver` | `langgraph-checkpoint-postgres` | PostgreSQL | 是 |

**关键发现：** 替换 `MemorySaver` 为 `SqliteSaver` 只需改一行代码，`thread_id` / `get_session_history` / `clear_session` 机制全部不用改，框架自动从 SQLite 读写。

### 双层存储架构

```
┌─────────────────────────────────────────────────────┐
│ MemoryService（新增, 统一接口, 全 async）             │
│                                                      │
│  读取: aget_context(session_id)                      │
│    1. 查 Redis → 摘要快照（长期上下文）               │
│    2. 查 SQLite → 最近 N 条消息（近期对话）           │
│    3. 拼装: 摘要 + 近期消息 → 返回                    │
│                                                      │
│  写入: 自动（LangGraph checkpointer 处理）            │
│  同步: await sync_summary(session_id)                │
│    → 从 SQLite 提取摘要 → 写入 Redis                  │
│  清空: await aclear(session_id)                      │
│    → 删 Redis 摘要 + 删 SQLite 线程                  │
├───────────────────────┬─────────────────────────────┤
│  SQLite 层（持久真相） │  Redis 层（快速上下文）       │
│                       │                             │
│  AsyncSqliteSaver     │  redis-py 同步客户端         │
│  (LangGraph 原生, aio)│                             │
│                       │  Key: session:{id}:summary   │
│  存储: 完整对话历史    │  Value: 摘要文本              │
│  方式: LangGraph 自动  │  TTL: 7天                    │
│  文件: ./data/chat.db │  降级: Redis不可用时跳过      │
└───────────────────────┴─────────────────────────────┘
```

### 三条核心数据流

**① 写入流程（用户发消息）**

```
用户发消息 → LangGraph agent.ainvoke()
  → SqliteSaver 自动写入完整消息到 SQLite（零侵入）
  → ainvoke 完成后 → MemoryService.sync_summary()
    → 从 SQLite 读最新消息列表
    → 检测 SummarizationMiddleware 生成的摘要 SystemMessage
    → 写入 Redis (session:{id}:summary)
```

**② 读取流程（新请求进来）**

```
新请求进来 → MemoryService.aget_context(session_id)
  → 查 Redis: session:{id}:summary → 拿到摘要（快速, ~1ms）
  → 查 SQLite: await AsyncSqliteSaver.aget_tuple(thread_id) → 拿到最近 N 条消息
  → 拼装: [摘要 SystemMessage] + [最近 N 条消息]
  → 交给 LangGraph agent 处理
```

> ⚠️ **重要：必须使用 async API**。`AsyncSqliteSaver` 的同步 `get()` 在事件循环线程中调用会
> 死锁（其内部 async 操作需要事件循环，而事件循环正被当前同步调用阻塞）。曾尝试用
> `ThreadPoolExecutor` 在子线程同步调用 `get()` 仍会死锁，最终改为全链路 async
>（`aget_tuple` / `adelete_thread`）。详见下方「实现踩坑」章节。

**③ 降级策略（Redis 不可用）**

```
Redis 连接失败 → 记录 warning 日志
  → 跳过摘要层, 纯 SQLite 读取（等价于只有 SqliteSaver）
  → 核心对话功能不受影响
  → Redis 恢复后自动恢复摘要层
```

## 组件接口设计

### MemoryService（新增）

> 注意：由于 `AsyncSqliteSaver` 只能通过 async API 安全调用（同步 `get()` 会在事件循环
> 线程死锁），MemoryService 中所有「读 checkpointer / 删线程」的方法都是 `async`；
> Redis 摘要读写是 redis-py 同步操作（毫秒级，无需 async），保持同步方法即可。

```python
class MemoryService:
    """双层记忆服务：SQLite 持久真相 + Redis 摘要快照"""

    def __init__(self, checkpointer: BaseCheckpointSaver):
        self.checkpointer = checkpointer   # AsyncSqliteSaver 实例（降级时为 MemorySaver）
        self._redis: Optional[redis.Redis] = None
        self._redis_available: bool = False

    # ---- Redis 摘要层（同步，毫秒级，无需 async）----
    def init_redis(self) -> None: ...          # 失败时 _redis_available=False（降级）
    def close_redis(self) -> None: ...
    def _get_summary(self, session_id: str) -> Optional[str]: ...   # 失败返回 None
    def _save_summary(self, session_id: str, summary: str) -> None: # 失败仅记日志

    # ---- SQLite checkpointer 层（必须 async，避免死锁）----
    async def _aget_messages(self, session_id: str) -> list[BaseMessage]:
        """await self.checkpointer.aget_tuple(config) → checkpoint.channel_values.messages"""

    def _extract_summary(self, messages: list) -> Optional[str]:
        """提取 SummarizationMiddleware 插入的摘要 SystemMessage（排除原始系统提示词）"""

    # ---- 统一对外接口（全 async）----
    async def aget_context(self, session_id: str) -> list[BaseMessage]:
        """Redis 摘要 + SQLite 近期消息（拼装上下文）"""

    async def sync_summary(self, session_id: str) -> None:
        """ainvoke 完成后：从 SQLite 提取摘要 → 写入 Redis"""

    async def aget_history(self, session_id: str) -> list[dict]:
        """获取会话历史（前端展示用，跳过 SystemMessage）"""

    async def aclear(self, session_id: str) -> bool:
        """清空双层存储：删 Redis 摘要 + await adelete_thread(session_id)"""
```

### 摘要提取逻辑

SummarizationMiddleware 压缩后会在消息列表头部插入一个 SystemMessage，内容是旧对话的摘要。提取逻辑：

```python
def _extract_summary(self, messages: list) -> str | None:
    """提取摘要：找到第一个 SystemMessage 且不是原始系统提示词"""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            content = msg.content
            # 原始系统提示词以"你是一个专业的AI助手"开头
            # 摘要 SystemMessage 是 SummarizationMiddleware 生成的
            if not content.startswith("你是一个专业的AI助手"):
                return content
    return None
```

## 文件改动清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `app/config.py` | 修改 | 新增 `sqlite_db_path`、`redis_url`、`redis_summary_ttl` 配置项 |
| `app/services/memory_service.py` | **新增** | MemoryService 双层记忆服务（全 async） |
| `app/services/rag_agent_service.py` | 修改 | 替换 MemorySaver 为 AsyncSqliteSaver；集成 MemoryService；query/query_stream 后 `await sync_summary`；`get_session_history`/`clear_session` 改为 `aget_session_history`/`aclear_session`（async，内部先调 `_initialize_agent` 确保 checkpointer 已就绪） |
| `app/api/chat.py` | 修改 | `get_session_info`/`clear_session` 路由改为 `await` 异步方法 |
| `app/main.py` | 修改 | lifespan 中检查 Redis 状态，关闭时调用 `rag_agent_service.cleanup()` 清理 SQLite/Redis |
| `.env` | 修改 | 新增 SQLITE_DB_PATH、REDIS_URL、REDIS_SUMMARY_TTL 环境变量 |
| `pyproject.toml` | 修改 | 新增 `langgraph-checkpoint-sqlite`、`redis`、`aiosqlite` 依赖 |

### 不改动的文件

| 文件 | 理由 |
|---|---|
| `app/agent/aiops/*` | AIOps 流程不使用 checkpointer，不受影响 |
| `app/tools/*` | 工具层与记忆无关 |
| `app/core/milvus_client.py` | 向量库与对话记忆无关 |

## 配置项设计

### config.py 新增

```python
# SQLite 持久化配置
sqlite_db_path: str = "./data/chat.db"

# Redis 摘要层配置
redis_url: str = "redis://localhost:6379/0"
redis_summary_ttl: int = 604800  # 7天（秒）
```

### .env 新增

```env
# 对话记忆持久化
SQLITE_DB_PATH=./data/chat.db
REDIS_URL=redis://localhost:6379/0
REDIS_SUMMARY_TTL=604800
```

## 关键设计决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| SQLite 文件路径 | `./data/chat.db` | 与 logs/ 同级，开发简单 |
| Redis Key 设计 | `session:{id}:summary` | 语义清晰，便于按 session 清理 |
| Redis TTL | 7 天 | 对话摘要时效性合理，过期自动清理 |
| 摘要写入时机 | `ainvoke` 完成后 | 不侵入 SummarizationMiddleware 内部，解耦 |
| 降级策略 | Redis 失败跳过 | 保证核心功能可用 |
| 消息窗口 | 最近 6 条（3 轮） | 复用现有 `trim_messages_middleware` 逻辑 |
| SQLite 模式 | 异步 `AsyncSqliteSaver` | 与项目 async 架构一致 |

## 降级与容错

| 故障场景 | 处理方式 |
|---|---|
| Redis 连接失败 | 跳过摘要层，纯 SQLite 读取，记录 warning |
| SQLite 文件损坏 | 捕获异常，降级为 MemorySaver，记录 error |
| 摘要提取失败 | 跳过 Redis 写入，不影响对话流程 |
| Redis TTL 过期 | 下次 sync_summary 时重新写入 |

## 测试策略

| 测试项 | 验证点 |
|---|---|
| 服务重启后恢复历史 | 重启后 `aget_session_history` 仍能返回历史消息 |
| 摘要写入 Redis | 对话超过 70% 阈值后，Redis 中出现摘要 |
| 摘要读取加速 | 新请求进来时，Redis 摘要被正确拼入上下文 |
| Redis 降级 | 断开 Redis 后，对话功能正常（纯 SQLite） |
| 清空会话 | aclear 后 SQLite 和 Redis 均无残留 |
| 多会话隔离 | 不同 session_id 的记忆互不干扰 |

## 实现踩坑记录

实施过程中遇到两个导致「服务卡死 / 历史读不出」的关键问题，记录如下，供后续维护参考。

### 踩坑 1：AsyncSqliteSaver 同步调用死锁（服务卡死无响应）

**现象：** 服务启动正常，但首次调用 `GET /api/chat/session/{id}` 或 `POST /api/chat` 后，
服务彻底无响应（`curl` 超时，exit code 28），日志无任何错误输出。

**根因：** `AsyncSqliteSaver` 是异步 checkpointer，其同步 `get()` 方法内部仍要运行 async
代码，需要事件循环。当在 FastAPI 的 async 路由线程里同步调用 `get()` 时：
- 同步调用阻塞了当前事件循环线程；
- 而 `get()` 内部的 async 操作又需要这个被阻塞的事件循环来调度；
- → 互相等待 → 死锁。

**走过的弯路：** 曾尝试用 `concurrent.futures.ThreadPoolExecutor` 在子线程中同步调用
`get()`（`AsyncSqliteSaver` 报错提示「Synchronous calls ... are only allowed from a
different thread」）。换了线程后不再报错，但子线程里没有可用的事件循环，依然死锁。

**正确修复：** 全链路改为 async，使用 `AsyncSqliteSaver` 原生的 async API：

| 旧（死锁） | 新（正确） |
|---|---|
| `self.checkpointer.get(config)` | `await self.checkpointer.aget_tuple(config)` |
| `self.checkpointer.delete_thread(tid)` | `await self.checkpointer.adelete_thread(tid)` |
| `def _get_messages` | `async def _aget_messages` |
| `def get_context` | `async def aget_context` |
| `def sync_summary` | `async def sync_summary` |
| `def get_history` | `async def aget_history` |
| `def clear` | `async def aclear` |

调用方相应改为 `await`：
- `rag_agent_service.query/query_stream`：`await mem_service.sync_summary(session_id)`
- `rag_agent_service.aget_session_history / aclear_session`：async 方法
- `api/chat.py` 路由：`await rag_agent_service.aget_session_history(...)`

> Redis 摘要读写（`_get_summary`/`_save_summary`）保持同步：redis-py 是同步客户端，
> 操作在毫秒级，不会阻塞事件循环，无需 async。

### 踩坑 2：查询历史时 checkpointer 未初始化（message_count 永远为 0）

**现象：** 修复死锁后，`GET /api/chat/session/{id}` 不再卡死，但返回 `message_count: 0`，
而直接用 `sqlite3` 查 `data/chat.db` 明明有 8 条 checkpoint 记录。

**根因：** `RagAgentService.__init__` 中 `self.checkpointer = MemorySaver()` 只是临时占位，
真正的 `AsyncSqliteSaver` 在 `_initialize_agent()` 中才替换；`MemoryService` 也在
`_initialize_agent()` 末尾才初始化。而 `get_session_info` 路由直接调用
`aget_session_history`，没有先触发 `_initialize_agent()`，导致：
- `get_memory_service()` 返回 `None`（未初始化）→ 走降级路径；
- 降级路径用的 `self.checkpointer` 还是空的 `MemorySaver` → 查不到 SQLite 数据。

**修复：** 在 `aget_session_history` 和 `aclear_session` 方法开头加上
`await self._initialize_agent()`（内部有 `if self._agent_initialized: return` 幂等保护），
确保 checkpointer 已替换为 `AsyncSqliteSaver`、`MemoryService` 已就绪后再读写。

### 验证结果

修复后端到端验证通过：
1. SQLite `data/chat.db` 中 `test-persist-1` 有 8 个 checkpoints；
2. 服务重启后 `GET /api/chat/session/test-persist-1` 返回 4 条历史消息；
3. 发送 "Do you still remember my name?"，Agent 回答 "your name is Xiao Ming"；
4. 对话后 `message_count` 从 4 → 6，新消息正确追加到 SQLite。
