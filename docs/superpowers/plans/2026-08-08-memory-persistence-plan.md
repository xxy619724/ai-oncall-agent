# 对话记忆持久化功能实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将对话记忆从纯内存（MemorySaver）持久化到 SQLite + Redis 双层存储，服务重启后可恢复历史对话。

**Architecture:** 用 LangGraph 原生 SqliteSaver 替换 MemorySaver（持久真相），新增 MemoryService 封装 Redis 摘要层（快速上下文），ainvoke 完成后自动同步摘要到 Redis，Redis 不可用时降级为纯 SQLite。

**Tech Stack:** LangGraph SqliteSaver（langgraph-checkpoint-sqlite）、redis（Python 客户端）、SQLite3

---

## ⚠️ 实施修订说明（2026-08-08，最终实现与下方原计划的差异）

下方 Task 3/4/5 的代码是初版计划，实际实施时为解决 **AsyncSqliteSaver 同步调用死锁**
问题，做了全链路 async 改造。最终代码以源文件为准，关键差异如下（原理与踩坑过程详见
[设计文档 - 实现踩坑记录](../specs/2026-08-08-memory-persistence-design.md#实现踩坑记录)）：

1. **checkpointer 改用 `AsyncSqliteSaver`（非同步 `SqliteSaver`）**
   - `import aiosqlite` + `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`
   - `self._sqlite_conn = await aiosqlite.connect(...)` → `AsyncSqliteSaver(conn)` → `await setup()`
   - 初始化放在 `_initialize_agent()`（async）中，`__init__` 仍用 `MemorySaver()` 临时占位。

2. **MemoryService 全 async**（原计划为同步方法）：
   | 原计划（同步） | 最终实现（async） |
   |---|---|
   | `_get_messages` → `checkpointer.get()` | `async _aget_messages` → `await checkpointer.aget_tuple()` |
   | `get_context` | `async aget_context` |
   | `sync_summary` | `async sync_summary` |
   | `get_history` | `async aget_history` |
   | `clear` → `delete_thread()` | `async aclear` → `await adelete_thread()` |
   - Redis 摘要读写（`_get_summary`/`_save_summary`）保持同步（redis-py，毫秒级）。

3. **RagAgentService 方法改名并 async 化**：
   - `get_session_history` → `async aget_session_history`，`clear_session` → `async aclear_session`
   - 两者方法体开头必须 `await self._initialize_agent()`（幂等），否则 checkpointer 还是
     空的 MemorySaver、MemoryService 未初始化，会读到 `message_count: 0`。
   - `query` / `query_stream` 中 `mem_service.sync_summary(...)` 改为 `await mem_service.sync_summary(...)`。
   - `cleanup` 中 SQLite 关闭改为 `await self._sqlite_conn.close()`。

4. **`app/api/chat.py` 也需改动**（原计划列为「不改动」）：
   - `get_session_info`：`history = await rag_agent_service.aget_session_history(session_id)`
   - `clear_session`：`success = await rag_agent_service.aclear_session(request.session_id)`

5. **依赖新增 `aiosqlite`**（原计划只有 `langgraph-checkpoint-sqlite` + `redis`）。

> 下方各 Task 的步骤结构（文件路径、验证命令、commit 节奏）仍然有效，仅代码片段需按
> 上述差异替换为 async 版本。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `app/services/memory_service.py`（新增） | 双层记忆服务：Redis 摘要读写 + SQLite 消息读取 + 摘要提取 + 统一接口 |
| `app/config.py`（修改） | 新增 sqlite_db_path / redis_url / redis_summary_ttl 配置项 |
| `app/services/rag_agent_service.py`（修改） | 替换 MemorySaver 为 SqliteSaver；集成 MemoryService；query/query_stream 后调 sync_summary |
| `app/main.py`（修改） | lifespan 中初始化/关闭 Redis 连接 |
| `.env`（修改） | 新增 SQLITE_DB_PATH / REDIS_URL / REDIS_SUMMARY_TTL |
| `pyproject.toml`（修改） | 新增 langgraph-checkpoint-sqlite / redis 依赖 |

---

## Task 1: 安装依赖

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env`

- [ ] **Step 1: 安装 langgraph-checkpoint-sqlite 和 redis 包**

Run:
```bash
uv add langgraph-checkpoint-sqlite redis
```
Expected: 两个包安装成功，pyproject.toml 自动更新

- [ ] **Step 2: 验证导入可用**

Run:
```bash
python -c "from langgraph.checkpoint.sqlite import SqliteSaver; import redis; print('OK')"
```
Expected: 输出 `OK`

- [ ] **Step 3: 修改 .env 新增环境变量**

在 `.env` 文件末尾追加：

```env

# 对话记忆持久化
SQLITE_DB_PATH=./data/chat.db
REDIS_URL=redis://localhost:6379/0
REDIS_SUMMARY_TTL=604800
```

- [ ] **Step 4: 创建 data 目录**

Run:
```bash
mkdir -p data
```
Expected: `data/` 目录创建成功

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env
git commit -m "feat: add sqlite and redis dependencies for memory persistence"
```

---

## Task 2: 新增配置项

**Files:**
- Modify: `app/config.py:62-64`（在 Prometheus 配置后追加）

- [ ] **Step 1: 在 config.py 的 Settings 类中新增配置项**

在 `app/config.py` 的 `prometheus_request_timeout` 配置项之后（约第 64 行），追加：

```python
    # 对话记忆持久化配置
    sqlite_db_path: str = "./data/chat.db"
    redis_url: str = "redis://localhost:6379/0"
    redis_summary_ttl: int = 604800  # 7天（秒）
```

- [ ] **Step 2: 验证配置加载**

Run:
```bash
python -c "from app.config import config; print(f'sqlite={config.sqlite_db_path}, redis={config.redis_url}, ttl={config.redis_summary_ttl}')"
```
Expected: 输出 `sqlite=./data/chat.db, redis=redis://localhost:6379/0, ttl=604800`

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: add sqlite/redis config for memory persistence"
```

---

## Task 3: 新增 MemoryService

**Files:**
- Create: `app/services/memory_service.py`

- [ ] **Step 1: 创建 memory_service.py 文件**

创建 `app/services/memory_service.py`，完整内容如下：

```python
"""双层记忆服务 - SQLite 持久真相 + Redis 摘要快照

提供统一的记忆管理接口：
- 读取: Redis 摘要（快速上下文）+ SQLite 近期消息（完整历史）
- 写入: LangGraph checkpointer 自动写入 SQLite，摘要通过 sync_summary 同步到 Redis
- 清空: 双层清空
- 降级: Redis 不可用时自动跳过，纯 SQLite 可用
"""

from typing import Any, Optional

import redis
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from loguru import logger

from app.config import config


class MemoryService:
    """双层记忆服务：SQLite 持久真相 + Redis 摘要快照"""

    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        """初始化记忆服务

        Args:
            checkpointer: LangGraph checkpointer 实例（SqliteSaver 或 MemorySaver）
        """
        self.checkpointer = checkpointer
        self._redis: Optional[redis.Redis] = None
        self._redis_available = False

    def init_redis(self) -> None:
        """初始化 Redis 连接，失败时标记不可用（降级模式）"""
        try:
            self._redis = redis.from_url(
                config.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # 测试连接
            self._redis.ping()
            self._redis_available = True
            logger.info(f"Redis 连接成功: {config.redis_url}")
        except Exception as e:
            self._redis = None
            self._redis_available = False
            logger.warning(f"Redis 连接失败，降级为纯 SQLite 模式: {e}")

    def close_redis(self) -> None:
        """关闭 Redis 连接"""
        if self._redis is not None:
            try:
                self._redis.close()
                logger.info("Redis 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭 Redis 连接失败: {e}")

    def _get_summary(self, session_id: str) -> Optional[str]:
        """从 Redis 获取摘要，失败时返回 None（降级）

        Args:
            session_id: 会话 ID

        Returns:
            摘要文本，或 None（无摘要 / Redis 不可用）
        """
        if not self._redis_available or self._redis is None:
            return None

        try:
            key = f"session:{session_id}:summary"
            value = self._redis.get(key)
            if value:
                logger.debug(f"从 Redis 获取摘要: {session_id}, 长度={len(value)}")
                return value
            return None
        except Exception as e:
            logger.warning(f"从 Redis 获取摘要失败: {e}")
            return None

    def _save_summary(self, session_id: str, summary: str) -> None:
        """保存摘要到 Redis，失败时记录日志（降级）

        Args:
            session_id: 会话 ID
            summary: 摘要文本
        """
        if not self._redis_available or self._redis is None:
            return

        try:
            key = f"session:{session_id}:summary"
            self._redis.setex(key, config.redis_summary_ttl, summary)
            logger.debug(f"摘要已保存到 Redis: {session_id}, 长度={len(summary)}")
        except Exception as e:
            logger.warning(f"保存摘要到 Redis 失败: {e}")

    def _get_messages(self, session_id: str) -> list[BaseMessage]:
        """从 checkpointer 获取会话消息列表

        Args:
            session_id: 会话 ID（即 thread_id）

        Returns:
            消息列表，无历史时返回空列表
        """
        try:
            config_dict = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = self.checkpointer.get(config_dict)

            if not checkpoint_tuple:
                return []

            # 安全提取 checkpoint 数据
            if hasattr(checkpoint_tuple, "checkpoint"):
                checkpoint_data = checkpoint_tuple.checkpoint
            else:
                checkpoint_data = (
                    checkpoint_tuple[0] if checkpoint_tuple else {}
                )

            if not isinstance(checkpoint_data, dict):
                return []

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            return messages if isinstance(messages, list) else []
        except Exception as e:
            logger.error(f"从 checkpointer 获取消息失败: {session_id}, 错误: {e}")
            return []

    def _extract_summary(self, messages: list[BaseMessage]) -> Optional[str]:
        """从消息列表中提取 SummarizationMiddleware 生成的摘要

        SummarizationMiddleware 压缩后会在消息列表头部插入一个 SystemMessage，
        内容是旧对话的摘要。通过排除原始系统提示词来识别摘要消息。

        Args:
            messages: 消息列表

        Returns:
            摘要文本，或 None（无摘要）
        """
        system_messages = [
            msg for msg in messages if isinstance(msg, SystemMessage)
        ]

        # 如果只有 1 个或 0 个 SystemMessage，说明没有摘要
        if len(system_messages) <= 1:
            return None

        # 原始系统提示词通常是第一个 SystemMessage
        # 摘要是 SummarizationMiddleware 额外插入的 SystemMessage
        for msg in system_messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            # 原始系统提示词以 "你是一个专业的AI助手" 开头
            if not str(content).startswith("你是一个专业的AI助手"):
                return str(content)

        return None

    def get_context(self, session_id: str) -> list[BaseMessage]:
        """获取会话上下文：Redis 摘要 + SQLite 近期消息

        读取流程：
        1. 查 Redis 摘要（快速, ~1ms）
        2. 查 SQLite 完整历史消息
        3. 拼装：[摘要 SystemMessage] + [历史消息]

        Args:
            session_id: 会话 ID

        Returns:
            拼装后的消息列表
        """
        # 1. 查 Redis 摘要
        summary = self._get_summary(session_id)

        # 2. 查 SQLite 消息
        messages = self._get_messages(session_id)

        # 3. 拼装
        if summary:
            return [SystemMessage(content=summary)] + messages
        return messages

    def sync_summary(self, session_id: str) -> None:
        """ainvoke 完成后调用：从 SQLite 提取摘要写入 Redis

        从 checkpointer 读取最新消息列表，检测 SummarizationMiddleware
        生成的摘要 SystemMessage，写入 Redis。

        Args:
            session_id: 会话 ID
        """
        messages = self._get_messages(session_id)
        if not messages:
            return

        summary = self._extract_summary(messages)
        if summary:
            self._save_summary(session_id, summary)

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """获取会话历史（前端展示用）

        从 checkpointer 读取消息，转换为前端格式。

        Args:
            session_id: 会话 ID

        Returns:
            消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        messages = self._get_messages(session_id)
        history: list[dict[str, Any]] = []

        from datetime import datetime

        for msg in messages:
            # 跳过系统消息（包括原始提示词和摘要）
            if isinstance(msg, SystemMessage):
                continue

            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)

            timestamp = getattr(msg, "timestamp", None)
            if not timestamp:
                timestamp = datetime.now().isoformat()

            history.append(
                {"role": role, "content": content, "timestamp": timestamp}
            )

        logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
        return history

    def clear(self, session_id: str) -> bool:
        """清空双层存储：删 Redis 摘要 + 删 SQLite 线程

        Args:
            session_id: 会话 ID

        Returns:
            是否成功
        """
        success = True

        # 1. 删 Redis 摘要
        if self._redis_available and self._redis is not None:
            try:
                key = f"session:{session_id}:summary"
                self._redis.delete(key)
                logger.debug(f"已删除 Redis 摘要: {session_id}")
            except Exception as e:
                logger.warning(f"删除 Redis 摘要失败: {e}")
                success = False

        # 2. 删 SQLite 线程
        try:
            # 尝试用 checkpointer 的 delete_thread 方法（MemorySaver 支持）
            if hasattr(self.checkpointer, "delete_thread"):
                self.checkpointer.delete_thread(session_id)
            else:
                # SqliteSaver 可能没有 delete_thread，用 SQL 删除
                logger.warning(
                    f"checkpointer 不支持 delete_thread，"
                    f"SQLite 线程数据需手动清理: {session_id}"
                )
            logger.info(f"已清除会话历史: {session_id}")
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            success = False

        return success


# 全局单例（延迟初始化，需要在外部注入 checkpointer）
memory_service: Optional[MemoryService] = None


def init_memory_service(checkpointer: BaseCheckpointSaver) -> MemoryService:
    """初始化全局 MemoryService 单例

    Args:
        checkpointer: LangGraph checkpointer 实例

    Returns:
        MemoryService 实例
    """
    global memory_service
    memory_service = MemoryService(checkpointer)
    memory_service.init_redis()
    return memory_service


def get_memory_service() -> Optional[MemoryService]:
    """获取全局 MemoryService 实例

    Returns:
        MemoryService 实例，未初始化时返回 None
    """
    return memory_service
```

- [ ] **Step 2: 验证导入**

Run:
```bash
python -c "from app.services.memory_service import MemoryService, init_memory_service, get_memory_service; print('OK')"
```
Expected: 输出 `OK`

- [ ] **Step 3: Commit**

```bash
git add app/services/memory_service.py
git commit -m "feat: add MemoryService for dual-layer memory (SQLite + Redis)"
```

---

## Task 4: 修改 rag_agent_service.py - 替换 MemorySaver

**Files:**
- Modify: `app/services/rag_agent_service.py`

- [ ] **Step 1: 替换 import**

在 `app/services/rag_agent_service.py` 第 17 行，替换：

```python
from langgraph.checkpoint.memory import MemorySaver
```

为：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
```

- [ ] **Step 2: 替换 checkpointer 初始化**

在 `app/services/rag_agent_service.py` 的 `__init__` 方法中（约第 109-110 行），替换：

```python
        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()
```

为：

```python
        # 创建持久化检查点：优先 SqliteSaver，失败降级为 MemorySaver
        try:
            self._sqlite_conn = sqlite3.connect(
                config.sqlite_db_path, check_same_thread=False
            )
            self.checkpointer = SqliteSaver(self._sqlite_conn)
            self.checkpointer.setup()
            logger.info(f"SqliteSaver 初始化成功: {config.sqlite_db_path}")
        except Exception as e:
            logger.error(f"SqliteSaver 初始化失败，降级为 MemorySaver: {e}")
            self._sqlite_conn = None
            self.checkpointer = MemorySaver()
```

- [ ] **Step 3: 添加 MemoryService 初始化**

在 `_initialize_agent` 方法中（约第 163 行，`self._agent_initialized = True` 之前），追加：

```python
        # 初始化 MemoryService（双层记忆服务）
        from app.services.memory_service import init_memory_service
        init_memory_service(self.checkpointer)
```

- [ ] **Step 4: 验证服务启动**

Run:
```bash
python -c "
import asyncio
from app.services.rag_agent_service import rag_agent_service
async def test():
    await rag_agent_service._initialize_agent()
    print('Agent 初始化成功')
asyncio.run(test())
"
```
Expected: 输出 `Agent 初始化成功`，日志中看到 `SqliteSaver 初始化成功`

- [ ] **Step 5: Commit**

```bash
git add app/services/rag_agent_service.py
git commit -m "feat: replace MemorySaver with SqliteSaver, init MemoryService"
```

---

## Task 5: 修改 rag_agent_service.py - 集成 MemoryService

**Files:**
- Modify: `app/services/rag_agent_service.py`

- [ ] **Step 1: 在 query 方法中添加 sync_summary 调用**

在 `query` 方法中（约第 249 行，`return answer` 之前），追加：

```python
                # 同步摘要到 Redis
                from app.services.memory_service import get_memory_service
                mem_service = get_memory_service()
                if mem_service:
                    mem_service.sync_summary(session_id)
```

- [ ] **Step 2: 在 query_stream 方法中添加 sync_summary 调用**

在 `query_stream` 方法中（约第 319 行，`yield {"type": "complete"}` 之前），追加：

```python
            # 同步摘要到 Redis
            from app.services.memory_service import get_memory_service
            mem_service = get_memory_service()
            if mem_service:
                mem_service.sync_summary(session_id)
```

- [ ] **Step 3: 委托 get_session_history 给 MemoryService**

将 `get_session_history` 方法（约第 329-392 行）替换为：

```python
    def get_session_history(self, session_id: str) -> list:
        """获取会话历史（委托给 MemoryService）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        from app.services.memory_service import get_memory_service

        mem_service = get_memory_service()
        if mem_service:
            return mem_service.get_history(session_id)

        # 降级：直接从 checkpointer 读取
        return self._get_history_from_checkpointer(session_id)

    def _get_history_from_checkpointer(self, session_id: str) -> list:
        """直接从 checkpointer 获取历史（降级方法）"""
        try:
            config = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = self.checkpointer.get(config)

            if not checkpoint_tuple:
                return []

            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint
            else:
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                timestamp = getattr(msg, 'timestamp', None)
                if not timestamp:
                    from datetime import datetime
                    timestamp = datetime.now().isoformat()
                history.append({"role": role, "content": content, "timestamp": timestamp})

            logger.info(f"获取会话历史(降级): {session_id}, 消息数量: {len(history)}")
            return history
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []
```

- [ ] **Step 4: 委托 clear_session 给 MemoryService**

将 `clear_session` 方法（约第 394-413 行）替换为：

```python
    def clear_session(self, session_id: str) -> bool:
        """清空会话历史（委托给 MemoryService）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        from app.services.memory_service import get_memory_service

        mem_service = get_memory_service()
        if mem_service:
            return mem_service.clear(session_id)

        # 降级：直接从 checkpointer 删除
        try:
            if hasattr(self.checkpointer, "delete_thread"):
                self.checkpointer.delete_thread(session_id)
            logger.info(f"已清除会话历史(降级): {session_id}")
            return True
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False
```

- [ ] **Step 5: 在 cleanup 方法中关闭 SQLite 连接**

在 `cleanup` 方法中（约第 415-422 行），追加 SQLite 和 Redis 关闭逻辑：

```python
    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理

            # 关闭 MemoryService Redis 连接
            from app.services.memory_service import get_memory_service
            mem_service = get_memory_service()
            if mem_service:
                mem_service.close_redis()

            # 关闭 SQLite 连接
            if hasattr(self, "_sqlite_conn") and self._sqlite_conn:
                self._sqlite_conn.close()
                logger.info("SQLite 连接已关闭")

            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
```

- [ ] **Step 6: 验证导入和基本功能**

Run:
```bash
python -c "
from app.services.rag_agent_service import rag_agent_service
print('checkpointer:', type(rag_agent_service.checkpointer).__name__)
print('OK')
"
```
Expected: 输出 `checkpointer: SqliteSaver` 和 `OK`

- [ ] **Step 7: Commit**

```bash
git add app/services/rag_agent_service.py
git commit -m "feat: integrate MemoryService into RagAgentService (sync_summary, delegate history/clear)"
```

---

## Task 6: 修改 main.py - Redis 生命周期管理

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: 在 lifespan 中添加 Redis 初始化提示**

在 `app/main.py` 的 `lifespan` 函数中（约第 33 行，`logger.info("=" * 60)` 之前），追加：

```python
    # MemoryService Redis 连接状态（实际初始化在 RagAgentService._initialize_agent 中）
    from app.services.memory_service import get_memory_service
    mem_service = get_memory_service()
    if mem_service and mem_service._redis_available:
        logger.info("✅ Redis 摘要层已就绪")
    else:
        logger.warning("⚠️ Redis 摘要层未就绪（降级为纯 SQLite 模式）")
```

- [ ] **Step 2: 在 lifespan 关闭时添加资源清理**

在 `lifespan` 函数的关闭部分（约第 39-41 行），修改为：

```python
    # 关闭时执行
    logger.info("🔌 正在关闭服务...")
    
    # 清理 RAG Agent 资源（SQLite + Redis）
    from app.services.rag_agent_service import rag_agent_service
    await rag_agent_service.cleanup()
    
    # 关闭 Milvus
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")
```

- [ ] **Step 3: 验证启动**

Run:
```bash
python -c "
import asyncio
from app.main import app
from app.config import config
print(f'App: {config.app_name}')
print('lifespan defined:', hasattr(app, 'router'))
print('OK')
"
```
Expected: 输出 `OK`

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: add Redis lifecycle management in lifespan"
```

---

## Task 7: 端到端验证

**Files:**
- 无文件修改，纯验证步骤

- [ ] **Step 1: 启动服务**

Run:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```
Expected: 日志中出现 `SqliteSaver 初始化成功` 和 `Redis 摘要层已就绪`（或降级警告）

- [ ] **Step 2: 发送对话验证写入**

Run:
```bash
curl -X POST http://localhost:9900/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"你好，我叫小明","id":"test-session-1"}'
```
Expected: 返回正常对话响应

- [ ] **Step 3: 查询会话历史验证持久化**

Run:
```bash
curl http://localhost:9900/api/chat/session/test-session-1
```
Expected: 返回包含 "你好，我叫小明" 的历史消息

- [ ] **Step 4: 重启服务验证持久化**

停止服务（Ctrl+C），重新启动：

Run:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

然后再次查询历史：

Run:
```bash
curl http://localhost:9900/api/chat/session/test-session-1
```
Expected: **重启后仍能返回历史消息**（核心验证点）

- [ ] **Step 5: 验证 SQLite 文件存在**

Run:
```bash
ls -la data/chat.db
```
Expected: 文件存在且大小 > 0

- [ ] **Step 6: 验证清空会话**

Run:
```bash
curl -X POST http://localhost:9900/api/chat/clear \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-session-1"}'
```
Expected: 返回 `会话已清空`

- [ ] **Step 7: 确认清空后历史为空**

Run:
```bash
curl http://localhost:9900/api/chat/session/test-session-1
```
Expected: `message_count: 0`

- [ ] **Step 8: 最终 Commit**

```bash
git add -A
git commit -m "feat: complete memory persistence (SQLite + Redis dual-layer)"
```

---

## 降级场景验证（可选）

- [ ] **Step 1: 停止 Redis 服务，验证降级**

停止 Redis（或修改 .env 中 REDIS_URL 为错误地址），重启服务，发送对话：

Run:
```bash
curl -X POST http://localhost:9900/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"测试降级模式","id":"test-degrade"}'
```
Expected: 对话正常返回，日志中出现 `Redis 连接失败，降级为纯 SQLite 模式`

- [ ] **Step 2: 验证降级模式下持久化仍可用**

重启服务后查询历史：

Run:
```bash
curl http://localhost:9900/api/chat/session/test-degrade
```
Expected: 历史消息仍存在（SQLite 持久化不受 Redis 影响）
