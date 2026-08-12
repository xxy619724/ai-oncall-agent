# Agent 可观测体系（Trace/Span/Metric）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 AIOps Agent 构建自研轻量级可观测体系，覆盖 Trace（全链路追踪）、Span（单步骤切片）、Metric（工具调用成功率/延迟/Token 聚合统计）三个核心维度，数据持久化到 SQLite，零外部依赖。

**Architecture:** 使用 `contextvars.ContextVar` 在 async 调用链中传播 trace_id，每个 LangGraph 节点通过 `@trace_node` 装饰器自动创建 Span（记录输入/输出/耗时/Token/状态），工具调用在 executor.py 层面统一埋点记录到 `tool_metrics` 表。观测数据独立存于 SQLite（`data/observability.db`），与业务 checkpoint 库隔离。无 trace 上下文时自动降级为零开销直通执行。

**Tech Stack:** Python 3.11 contextvars / aiosqlite（异步 SQLite）/ Loguru / FastAPI（查询端点）/ LangGraph（被观测对象）

---

## 文件结构

### 新建文件（5 个）

| 文件 | 职责 |
|------|------|
| `app/observability/__init__.py` | 模块导出 |
| `app/observability/store.py` | SQLite 存储层：traces/spans/tool_metrics 三张表 + 增删查 |
| `app/observability/trace.py` | TraceContext + Span 数据类 + contextvars 传播 + span 上下文管理器 |
| `app/observability/metrics.py` | MetricsCollector 单例：记录节点/工具指标 + 聚合查询 |
| `app/observability/instrumentation.py` | `@trace_node` 装饰器 + 状态截断/Token 提取工具函数 |

### 修改文件（6 个）

| 文件 | 改动 |
|------|------|
| `app/config.py` | 新增 observability 配置段（db 路径、开关、截断长度） |
| `app/agent/aiops/planner.py` | 加 `@trace_node("planner")` 一行 |
| `app/agent/aiops/executor.py` | 加 `@trace_node("executor")` + 工具调用埋点 |
| `app/agent/aiops/replanner.py` | 加 `@trace_node("replanner")` 一行 |
| `app/agent/aiops/memory_writer.py` | 加 `@trace_node("memory_writer")` 一行 |
| `app/services/aiops_service.py` | execute() 中创建/结束 trace，设置 contextvar |
| `app/main.py` | lifespan 中初始化 + 清理 observability store |
| `app/api/aiops.py` | 新增 GET /metrics、GET /traces/{trace_id} 端点 |

---

## 数据模型

### traces 表（一次 AIOps 执行 = 一条 trace）

```sql
CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,          -- UUID
    session_id      TEXT NOT NULL,             -- 会话 ID
    input_text      TEXT,                      -- 用户输入（截断 500 字符）
    status          TEXT DEFAULT 'running',    -- running/completed/failed
    started_at      TEXT NOT NULL,             -- ISO 时间戳
    ended_at        TEXT,                      -- ISO 时间戳
    total_duration_ms REAL,                    -- 总耗时（毫秒）
    total_tokens    INTEGER DEFAULT 0,         -- 全链路 Token 总量
    node_count      INTEGER DEFAULT 0,         -- 节点执行次数
    tool_call_count INTEGER DEFAULT 0,         -- 工具调用次数
    error_message   TEXT                       -- 失败原因（status=failed 时）
);
```

### spans 表（一次节点执行 = 一条 span）

```sql
CREATE TABLE IF NOT EXISTS spans (
    span_id         TEXT PRIMARY KEY,          -- UUID
    trace_id        TEXT NOT NULL,             -- 所属 trace
    node_name       TEXT NOT NULL,             -- planner/executor/replanner/memory_writer
    span_type       TEXT DEFAULT 'node',       -- node（预留 tool/llm/memory_op）
    input_summary   TEXT,                      -- JSON：截断后的输入状态
    output_summary  TEXT,                      -- JSON：截断后的输出状态
    status          TEXT DEFAULT 'running',    -- running/completed/failed
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    duration_ms     REAL,                      -- 单节点耗时（毫秒）
    token_usage     INTEGER DEFAULT 0,         -- 该节点 Token 消耗
    metadata        TEXT                       -- JSON：附加数据（工具名、决策动作等）
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_node_name ON spans(node_name);
```

### tool_metrics 表（一次工具调用 = 一条记录）

```sql
CREATE TABLE IF NOT EXISTS tool_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL,
    tool_name       TEXT NOT NULL,             -- retrieve_knowledge/query_prometheus_alerts/...
    node_name       TEXT NOT NULL,             -- 调用方节点（通常是 executor）
    success         INTEGER NOT NULL,          -- 1=成功 0=失败
    duration_ms     REAL NOT NULL,             -- 工具耗时（毫秒）
    token_usage     INTEGER DEFAULT 0,         -- 关联 Token（如有）
    called_at       TEXT NOT NULL,             -- ISO 时间戳
    error_message   TEXT                       -- 失败原因
);
CREATE INDEX IF NOT EXISTS idx_tool_metrics_tool_name ON tool_metrics(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_metrics_trace_id ON tool_metrics(trace_id);
```

---

## Task 1: 配置 + 存储层（config + store.py）

**Files:**
- Modify: `app/config.py`（在 `memory_default_ttl_days` 之后新增配置段）
- Create: `app/observability/__init__.py`
- Create: `app/observability/store.py`

- [ ] **Step 1: 在 config.py 新增可观测配置段**

在 `app/config.py` 的 `memory_default_ttl_days` 字段之后、`@property mcp_servers` 之前，插入：

```python
    # 可观测体系配置（Trace/Span/Metric）
    observability_enabled: bool = True            # 总开关：False 时所有埋点零开销直通
    observability_db_path: str = "./data/observability.db"  # 独立于 checkpoint 库
    observability_span_input_max_len: int = 500   # Span 输入摘要截断长度（字符）
    observability_span_output_max_len: int = 500  # Span 输出摘要截断长度（字符）
    observability_trace_input_max_len: int = 500  # Trace 输入截断长度（字符）
```

- [ ] **Step 2: 创建 app/observability/__init__.py**

```python
"""可观测体系模块 - Trace/Span/Metric 自研轻量实现

设计原则：
- 零外部依赖：不依赖 LangSmith/OpenTelemetry，数据存 SQLite
- 零侵入：通过 contextvars 传播 trace_id，不污染业务 State
- 零开销降级：observability_enabled=False 或无 trace 上下文时，埋点自动跳过
"""

from app.observability.store import observability_store, ObservabilityStore
from app.observability.trace import (
    TraceContext,
    Span,
    current_trace,
    start_trace,
    span,
)
from app.observability.metrics import metrics_collector, MetricsCollector
from app.observability.instrumentation import trace_node

__all__ = [
    "observability_store",
    "ObservabilityStore",
    "TraceContext",
    "Span",
    "current_trace",
    "start_trace",
    "span",
    "metrics_collector",
    "MetricsCollector",
    "trace_node",
]
```

- [ ] **Step 3: 创建 app/observability/store.py**

```python
"""可观测数据 SQLite 存储层

三张表：
- traces：一次 AIOps 执行的全链路记录
- spans：单节点执行切片
- tool_metrics：工具调用指标（用于聚合统计成功率/延迟）

所有方法均为 async，使用 aiosqlite 连接。
连接由 main.py lifespan 初始化，全局单例 observability_store。
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite
from loguru import logger

from app.config import config


class ObservabilityStore:
    """可观测数据 SQLite 存储（异步）"""

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or config.observability_db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化数据库连接 + 建表（在 main.py lifespan 中调用）"""
        if self._initialized:
            return
        try:
            self._conn = await aiosqlite.connect(self._db_path)
            await self._create_tables()
            self._initialized = True
            logger.info(f"ObservabilityStore 初始化成功: {self._db_path}")
        except Exception as e:
            logger.error(f"ObservabilityStore 初始化失败（可观测数据将无法持久化）: {e}")
            self._conn = None
            self._initialized = False  # 允许重试

    async def _create_tables(self) -> None:
        """建表（IF NOT EXISTS，幂等）"""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id         TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                input_text       TEXT,
                status           TEXT DEFAULT 'running',
                started_at       TEXT NOT NULL,
                ended_at         TEXT,
                total_duration_ms REAL,
                total_tokens     INTEGER DEFAULT 0,
                node_count       INTEGER DEFAULT 0,
                tool_call_count  INTEGER DEFAULT 0,
                error_message    TEXT
            );

            CREATE TABLE IF NOT EXISTS spans (
                span_id         TEXT PRIMARY KEY,
                trace_id        TEXT NOT NULL,
                node_name       TEXT NOT NULL,
                span_type       TEXT DEFAULT 'node',
                input_summary   TEXT,
                output_summary  TEXT,
                status          TEXT DEFAULT 'running',
                started_at      TEXT NOT NULL,
                ended_at        TEXT,
                duration_ms     REAL,
                token_usage     INTEGER DEFAULT 0,
                metadata        TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_spans_node_name ON spans(node_name);

            CREATE TABLE IF NOT EXISTS tool_metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id        TEXT NOT NULL,
                tool_name       TEXT NOT NULL,
                node_name       TEXT NOT NULL,
                success         INTEGER NOT NULL,
                duration_ms     REAL NOT NULL,
                token_usage     INTEGER DEFAULT 0,
                called_at       TEXT NOT NULL,
                error_message   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tool_metrics_tool_name ON tool_metrics(tool_name);
            CREATE INDEX IF NOT EXISTS idx_tool_metrics_trace_id ON tool_metrics(trace_id);
        """)
        await self._conn.commit()

    @property
    def available(self) -> bool:
        """存储是否可用（连接已建立且已初始化）"""
        return self._conn is not None and self._initialized

    # ============================================================
    # Trace 增删查
    # ============================================================

    async def save_trace(self, trace_id: str, session_id: str, input_text: str,
                         started_at: str) -> None:
        """创建 trace 记录（execute 开始时调用）"""
        if not self.available:
            return
        assert self._conn is not None
        truncated = input_text[:config.observability_trace_input_max_len]
        try:
            await self._conn.execute(
                """INSERT INTO traces (trace_id, session_id, input_text, status, started_at)
                   VALUES (?, ?, ?, 'running', ?)""",
                (trace_id, session_id, truncated, started_at),
            )
            await self._conn.commit()
        except Exception as e:
            logger.error(f"save_trace 失败: {e}")

    async def update_trace_status(self, trace_id: str, status: str,
                                  ended_at: str, total_duration_ms: float,
                                  total_tokens: int, node_count: int,
                                  tool_call_count: int,
                                  error_message: str = "") -> None:
        """更新 trace 状态（execute 结束/异常时调用）"""
        if not self.available:
            return
        assert self._conn is not None
        try:
            await self._conn.execute(
                """UPDATE traces
                   SET status = ?, ended_at = ?, total_duration_ms = ?,
                       total_tokens = ?, node_count = ?, tool_call_count = ?,
                       error_message = ?
                   WHERE trace_id = ?""",
                (status, ended_at, total_duration_ms, total_tokens,
                 node_count, tool_call_count, error_message, trace_id),
            )
            await self._conn.commit()
        except Exception as e:
            logger.error(f"update_trace_status 失败: {e}")

    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """查询单条 trace（含 spans + tool_metrics，用于全链路复现）"""
        if not self.available:
            return None
        assert self._conn is not None
        try:
            async with self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return None

            columns = [d[0] for d in cur.description]
            trace = dict(zip(columns, row))

            # 关联 spans
            async with self._conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at",
                (trace_id,),
            ) as cur:
                span_rows = await cur.fetchall()
                span_cols = [d[0] for d in cur.description]
            trace["spans"] = [dict(zip(span_cols, r)) for r in span_rows]

            # 关联 tool_metrics
            async with self._conn.execute(
                "SELECT * FROM tool_metrics WHERE trace_id = ? ORDER BY called_at",
                (trace_id,),
            ) as cur:
                tool_rows = await cur.fetchall()
                tool_cols = [d[0] for d in cur.description]
            trace["tool_metrics"] = [dict(zip(tool_cols, r)) for r in tool_rows]

            return trace
        except Exception as e:
            logger.error(f"get_trace 失败: {e}")
            return None

    async def list_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """列出最近的 trace（不含 spans 明细）"""
        if not self.available:
            return []
        assert self._conn is not None
        try:
            async with self._conn.execute(
                "SELECT * FROM traces ORDER BY started_at DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.error(f"list_traces 失败: {e}")
            return []

    # ============================================================
    # Span 增查
    # ============================================================

    async def save_span(self, span: Dict[str, Any]) -> None:
        """保存单条 span（节点执行结束时调用）"""
        if not self.available:
            return
        assert self._conn is not None
        try:
            await self._conn.execute(
                """INSERT INTO spans
                   (span_id, trace_id, node_name, span_type, input_summary,
                    output_summary, status, started_at, ended_at, duration_ms,
                    token_usage, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    span["span_id"], span["trace_id"], span["node_name"],
                    span.get("span_type", "node"),
                    span.get("input_summary"), span.get("output_summary"),
                    span.get("status", "completed"), span["started_at"],
                    span.get("ended_at"), span.get("duration_ms", 0),
                    span.get("token_usage", 0), span.get("metadata"),
                ),
            )
            await self._conn.commit()
        except Exception as e:
            logger.error(f"save_span 失败: {e}")

    # ============================================================
    # Tool Metric 增查
    # ============================================================

    async def save_tool_metric(self, trace_id: str, tool_name: str,
                               node_name: str, success: bool,
                               duration_ms: float, token_usage: int = 0,
                               error_message: str = "") -> None:
        """保存单条工具调用指标"""
        if not self.available:
            return
        assert self._conn is not None
        called_at = datetime.now().isoformat()
        try:
            await self._conn.execute(
                """INSERT INTO tool_metrics
                   (trace_id, tool_name, node_name, success, duration_ms,
                    token_usage, called_at, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (trace_id, tool_name, node_name, 1 if success else 0,
                 duration_ms, token_usage, called_at, error_message),
            )
            await self._conn.commit()
        except Exception as e:
            logger.error(f"save_tool_metric 失败: {e}")

    async def get_tool_metrics_summary(self) -> Dict[str, Any]:
        """聚合统计：各工具的调用次数/成功率/平均延迟"""
        if not self.available:
            return {}
        assert self._conn is not None
        try:
            async with self._conn.execute(
                """SELECT
                       tool_name,
                       COUNT(*) as total_calls,
                       SUM(success) as success_count,
                       AVG(duration_ms) as avg_duration_ms,
                       MAX(duration_ms) as max_duration_ms,
                       SUM(token_usage) as total_tokens
                   FROM tool_metrics
                   GROUP BY tool_name
                   ORDER BY total_calls DESC"""
            ) as cur:
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]

            tools = []
            for r in rows:
                d = dict(zip(cols, r))
                total = d["total_calls"] or 0
                success = d["success_count"] or 0
                tools.append({
                    "tool_name": d["tool_name"],
                    "total_calls": total,
                    "success_count": success,
                    "success_rate": round(success / total, 4) if total > 0 else 0,
                    "avg_duration_ms": round(d["avg_duration_ms"] or 0, 2),
                    "max_duration_ms": round(d["max_duration_ms"] or 0, 2),
                    "total_tokens": d["total_tokens"] or 0,
                })
            return {"tools": tools}
        except Exception as e:
            logger.error(f"get_tool_metrics_summary 失败: {e}")
            return {}

    async def get_node_metrics_summary(self) -> List[Dict[str, Any]]:
        """聚合统计：各节点的执行次数/平均耗时/Token"""
        if not self.available:
            return []
        assert self._conn is not None
        try:
            async with self._conn.execute(
                """SELECT
                       node_name,
                       COUNT(*) as total_runs,
                       SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as success_count,
                       AVG(duration_ms) as avg_duration_ms,
                       SUM(token_usage) as total_tokens
                   FROM spans
                   WHERE span_type = 'node'
                   GROUP BY node_name
                   ORDER BY total_runs DESC"""
            ) as cur:
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description]

            nodes = []
            for r in rows:
                d = dict(zip(cols, r))
                total = d["total_runs"] or 0
                success = d["success_count"] or 0
                nodes.append({
                    "node_name": d["node_name"],
                    "total_runs": total,
                    "success_count": success,
                    "success_rate": round(success / total, 4) if total > 0 else 0,
                    "avg_duration_ms": round(d["avg_duration_ms"] or 0, 2),
                    "total_tokens": d["total_tokens"] or 0,
                })
            return nodes
        except Exception as e:
            logger.error(f"get_node_metrics_summary 失败: {e}")
            return []

    async def cleanup(self) -> None:
        """关闭连接（在 main.py lifespan shutdown 中调用）"""
        if self._conn is not None:
            try:
                await self._conn.close()
                logger.info("ObservabilityStore 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭 ObservabilityStore 连接失败: {e}")
            finally:
                self._conn = None
                self._initialized = False


# 全局单例
observability_store = ObservabilityStore()
```

- [ ] **Step 4: 验证 store.py 语法**

Run: `python -c "from app.observability.store import observability_store; print('OK')"`
Expected: 输出 `OK`（无 ImportError / SyntaxError）

- [ ] **Step 5: 提交**

```bash
git add app/config.py app/observability/__init__.py app/observability/store.py
git commit -m "feat(observability): 新增配置项 + SQLite 存储层（traces/spans/tool_metrics 三表）"
```

---

## Task 2: TraceContext + Span 核心（trace.py）

**Files:**
- Create: `app/observability/trace.py`

**核心设计：** 使用 `contextvars.ContextVar` 在 async 调用链中传播 trace_id。LangGraph 的 `astream()` 在同一事件循环中顺序 await 各节点，contextvar 对所有子协程可见。节点无需修改签名即可读取当前 trace。

- [ ] **Step 1: 创建 app/observability/trace.py**

```python
"""TraceContext + Span 核心：基于 contextvars 的 async 链路传播

工作原理：
1. execute() 调用 start_trace() 设置 contextvar
2. LangGraph astream() 在同一事件循环顺序执行节点
3. 节点内的 @trace_node 装饰器通过 current_trace() 读取上下文
4. 无上下文时（如单元测试或 observability_enabled=False）自动降级为零开销直通

关键：contextvars 在 asyncio.create_task() 时会 copy 上下文，
因此子任务能读到 trace_id，但修改不会影响父任务（符合预期）。
"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from app.config import config
from app.observability.store import observability_store


# contextvar：当前激活的 trace 上下文（None 表示无追踪）
_current_trace: ContextVar[Optional["TraceContext"]] = ContextVar(
    "_current_trace", default=None
)


@dataclass
class TraceContext:
    """追踪上下文（一次 AIOps 执行对应一个）"""

    trace_id: str
    session_id: str
    input_text: str
    started_at: str
    # 运行期累计统计（execute 结束时写入 store）
    total_tokens: int = 0
    node_count: int = 0
    tool_call_count: int = 0


@dataclass
class Span:
    """单步骤切片数据（节点执行结束时持久化）"""

    span_id: str
    trace_id: str
    node_name: str
    span_type: str = "node"
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    status: str = "running"
    started_at: str = ""
    ended_at: Optional[str] = None
    duration_ms: float = 0.0
    token_usage: int = 0
    metadata: Optional[str] = None  # JSON 字符串


def current_trace() -> Optional[TraceContext]:
    """获取当前激活的 trace 上下文（无则返回 None）"""
    if not config.observability_enabled:
        return None
    return _current_trace.get()


def _truncate(text: str, max_len: int) -> str:
    """截断文本到指定长度，超长加省略号"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"


def _state_to_summary(state: Any, max_len: int) -> str:
    """将 LangGraph state（dict）截断为 JSON 摘要"""
    import json

    if state is None:
        return ""
    try:
        if isinstance(state, dict):
            # 只取关键字段，避免 plan/past_steps 过长
            safe = {}
            for k, v in state.items():
                if k == "input":
                    safe[k] = _truncate(str(v), 200)
                elif k == "plan":
                    # plan 是 List[str]，每条截断 100 字符
                    safe[k] = [_truncate(str(s), 100) for s in (v or [])][:5]
                elif k == "past_steps":
                    # past_steps 是 List[tuple]，只记数量和最后一条摘要
                    steps = v or []
                    safe[k] = {
                        "count": len(steps),
                        "last": _truncate(str(steps[-1]), 200) if steps else None,
                    }
                elif k == "response":
                    safe[k] = _truncate(str(v), 200)
                else:
                    safe[k] = _truncate(str(v), 100)
            return json.dumps(safe, ensure_ascii=False)
        return _truncate(str(state), max_len)
    except Exception:
        return _truncate(str(state), max_len)


class start_trace:
    """上下文管理器：启动一个 trace 并设置 contextvar

    用法：
        with start_trace(session_id, input_text) as trace:
            # 在此范围内的所有 async 调用都能通过 current_trace() 读到 trace
            await graph.astream(...)
    """

    def __init__(self, session_id: str, input_text: str):
        self.trace_id = str(uuid.uuid4())
        self.session_id = session_id
        self.input_text = input_text
        self.started_at = datetime.now().isoformat()
        self._ctx: TraceContext = TraceContext(
            trace_id=self.trace_id,
            session_id=session_id,
            input_text=input_text,
            started_at=self.started_at,
        )
        self._token = None  # contextvar token，用于 reset

    def __enter__(self) -> TraceContext:
        if not config.observability_enabled:
            return self._ctx
        # 设置 contextvar，保存 token 用于退出时恢复
        self._token = _current_trace.set(self._ctx)
        # 异步保存 trace 记录到 SQLite（fire-and-forget，不阻塞）
        import asyncio
        asyncio.ensure_future(
            observability_store.save_trace(
                self.trace_id, self.session_id, self.input_text, self.started_at
            )
        )
        logger.info(f"[Trace {self.trace_id[:8]}] 开始追踪: {self.input_text[:80]}")
        return self._ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not config.observability_enabled:
            return False
        # 恢复 contextvar
        if self._token is not None:
            _current_trace.reset(self._token)

        # 计算 trace 级统计并更新 store
        ended_at = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(self.started_at)
        total_duration_ms = (datetime.now() - start_dt).total_seconds() * 1000

        status = "completed" if exc_type is None else "failed"
        error_msg = f"{exc_type.__name__}: {exc_val}" if exc_val else ""

        import asyncio
        asyncio.ensure_future(
            observability_store.update_trace_status(
                trace_id=self.trace_id,
                status=status,
                ended_at=ended_at,
                total_duration_ms=round(total_duration_ms, 2),
                total_tokens=self._ctx.total_tokens,
                node_count=self._ctx.node_count,
                tool_call_count=self._ctx.tool_call_count,
                error_message=error_msg,
            )
        )
        logger.info(
            f"[Trace {self.trace_id[:8]}] 追踪结束: status={status}, "
            f"duration={total_duration_ms:.0f}ms, tokens={self._ctx.total_tokens}, "
            f"nodes={self._ctx.node_count}, tools={self._ctx.tool_call_count}"
        )
        return False  # 不吞异常
```

- [ ] **Step 2: 验证 trace.py 语法**

Run: `python -c "from app.observability.trace import start_trace, current_trace, Span; print('OK')"`
Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add app/observability/trace.py
git commit -m "feat(observability): TraceContext + Span + contextvars 异步链路传播"
```

---

## Task 3: MetricsCollector 聚合统计（metrics.py）

**Files:**
- Create: `app/observability/metrics.py`

**职责：** 运行期累计统计 + 持久化。每次节点/工具调用时记录，聚合查询委托给 store。

- [ ] **Step 1: 创建 app/observability/metrics.py**

```python
"""MetricsCollector 单例：运行期指标记录 + 聚合查询

记录维度：
- 节点级：执行次数/成功失败/耗时/Token
- 工具级：调用次数/成功率/延迟/Token

数据流：
  节点/工具执行 → MetricsCollector.record_xxx() → store.save_xxx() → SQLite
  查询时 → store.get_xxx_summary() → 聚合统计

无 trace 上下文时，record 方法自动跳过（零开销降级）。
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import config
from app.observability.store import observability_store
from app.observability.trace import current_trace


class MetricsCollector:
    """指标收集器（全局单例）"""

    async def record_node_completion(
        self,
        node_name: str,
        duration_ms: float,
        token_usage: int = 0,
        status: str = "completed",
    ) -> None:
        """记录节点执行完成（由 @trace_node 装饰器调用）"""
        if not config.observability_enabled:
            return
        trace_ctx = current_trace()
        if trace_ctx is None:
            return
        # 累计到 trace 上下文（execute 结束时统一写库）
        trace_ctx.node_count += 1
        trace_ctx.total_tokens += token_usage

    async def record_tool_call(
        self,
        tool_name: str,
        node_name: str,
        success: bool,
        duration_ms: float,
        token_usage: int = 0,
        error_message: str = "",
    ) -> None:
        """记录一次工具调用（由 executor.py 调用）"""
        if not config.observability_enabled:
            return
        trace_ctx = current_trace()
        if trace_ctx is None:
            return
        # 累计到 trace 上下文
        trace_ctx.tool_call_count += 1
        trace_ctx.total_tokens += token_usage
        # 持久化到 tool_metrics 表
        await observability_store.save_tool_metric(
            trace_id=trace_ctx.trace_id,
            tool_name=tool_name,
            node_name=node_name,
            success=success,
            duration_ms=duration_ms,
            token_usage=token_usage,
            error_message=error_message,
        )

    async def get_summary(self) -> Dict[str, Any]:
        """获取聚合统计摘要（供 /metrics 端点调用）"""
        tool_summary = await observability_store.get_tool_metrics_summary()
        node_summary = await observability_store.get_node_metrics_summary()
        traces = await observability_store.list_traces(limit=10)

        # 计算全局工具成功率
        all_tools = tool_summary.get("tools", [])
        total_calls = sum(t["total_calls"] for t in all_tools)
        total_success = sum(t["success_count"] for t in all_tools)
        global_success_rate = (
            round(total_success / total_calls, 4) if total_calls > 0 else 0
        )

        return {
            "tool_metrics": all_tools,
            "node_metrics": node_summary,
            "recent_traces": traces,
            "global": {
                "total_tool_calls": total_calls,
                "global_tool_success_rate": global_success_rate,
                "total_traces": len(traces),
            },
        }


# 全局单例
metrics_collector = MetricsCollector()
```

- [ ] **Step 2: 验证 metrics.py 语法**

Run: `python -c "from app.observability.metrics import metrics_collector; print('OK')"`
Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add app/observability/metrics.py
git commit -m "feat(observability): MetricsCollector 单例 - 节点/工具指标记录 + 聚合查询"
```

---

## Task 4: 节点 instrumentation 装饰器 + 应用到 4 个节点

**Files:**
- Create: `app/observability/instrumentation.py`
- Modify: `app/agent/aiops/planner.py`（加装饰器）
- Modify: `app/agent/aiops/executor.py`（加装饰器）
- Modify: `app/agent/aiops/replanner.py`（加装饰器）
- Modify: `app/agent/aiops/memory_writer.py`（加装饰器）

- [ ] **Step 1: 创建 app/observability/instrumentation.py**

```python
"""节点埋点装饰器 + 工具函数

@trace_node 装饰器工作流：
1. 读取 current_trace()，无上下文则直接执行原函数（零开销降级）
2. 创建 Span，记录 started_at + 输入摘要
3. 执行原节点函数
4. 记录输出摘要 + ended_at + duration_ms + token_usage
5. 异步持久化 Span 到 SQLite
6. 调用 MetricsCollector 记录节点指标
"""

import functools
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from loguru import logger

from app.config import config
from app.observability.metrics import metrics_collector
from app.observability.store import observability_store
from app.observability.trace import Span, current_trace, _state_to_summary, _truncate


def _extract_token_usage(result: Any) -> int:
    """从节点返回值中提取 Token 用量（如果有）"""
    # 节点返回的是 state dict，不直接含 token 信息
    # token 主要从 LLM response 提取，在 executor.py 中单独记录
    return 0


def trace_node(node_name: str):
    """LangGraph 节点埋点装饰器

    用法：
        @trace_node("planner")
        async def planner(state: PlanExecuteState) -> Dict[str, Any]:
            ...

    行为：
    - observability_enabled=False 或无 trace 上下文时，零开销直通
    - 有 trace 上下文时，创建 Span 记录输入/输出/耗时/状态
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(state: Any, *args, **kwargs) -> Dict[str, Any]:
            trace_ctx = current_trace()
            # 无 trace 上下文，直接执行（零开销降级）
            if trace_ctx is None:
                return await func(state, *args, **kwargs)

            # 创建 Span
            span_id = str(uuid.uuid4())
            started_at = datetime.now().isoformat()
            input_summary = _state_to_summary(
                state, config.observability_span_input_max_len
            )

            logger.debug(f"[Span {node_name}] 开始执行")

            status = "completed"
            output_summary: Optional[str] = None
            token_usage = 0
            error_message = ""

            try:
                result = await func(state, *args, **kwargs)
                output_summary = _state_to_summary(
                    result, config.observability_span_output_max_len
                )
                return result
            except Exception as e:
                status = "failed"
                error_message = str(e)
                logger.error(f"[Span {node_name}] 执行失败: {e}")
                raise
            finally:
                ended_at = datetime.now().isoformat()
                start_dt = datetime.fromisoformat(started_at)
                duration_ms = round(
                    (datetime.now() - start_dt).total_seconds() * 1000, 2
                )

                # 构建 Span 数据
                import json
                span_metadata = json.dumps(
                    {"error": error_message} if error_message else {},
                    ensure_ascii=False,
                )

                span_data = {
                    "span_id": span_id,
                    "trace_id": trace_ctx.trace_id,
                    "node_name": node_name,
                    "span_type": "node",
                    "input_summary": input_summary,
                    "output_summary": output_summary,
                    "status": status,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_ms": duration_ms,
                    "token_usage": token_usage,
                    "metadata": span_metadata,
                }

                # 异步持久化（不阻塞主流程）
                import asyncio
                asyncio.ensure_future(observability_store.save_span(span_data))
                asyncio.ensure_future(
                    metrics_collector.record_node_completion(
                        node_name=node_name,
                        duration_ms=duration_ms,
                        token_usage=token_usage,
                        status=status,
                    )
                )

                logger.debug(
                    f"[Span {node_name}] 结束: status={status}, "
                    f"duration={duration_ms}ms"
                )

        return wrapper
    return decorator
```

- [ ] **Step 2: 在 planner.py 应用装饰器**

在 `app/agent/aiops/planner.py` 中：

1. 在文件顶部 import 区（`from .utils import format_tools_description` 之后）新增：
```python
from app.observability import trace_node
```

2. 在 `async def planner(state: PlanExecuteState) -> Dict[str, Any]:` 上方加装饰器：
```python
@trace_node("planner")
async def planner(state: PlanExecuteState) -> Dict[str, Any]:
```

- [ ] **Step 3: 在 executor.py 应用装饰器**

在 `app/agent/aiops/executor.py` 中：

1. 在文件顶部 import 区新增：
```python
from app.observability import trace_node
```

2. 在 `async def executor(state: PlanExecuteState) -> Dict[str, Any]:` 上方加装饰器：
```python
@trace_node("executor")
async def executor(state: PlanExecuteState) -> Dict[str, Any]:
```

- [ ] **Step 4: 在 replanner.py 应用装饰器**

在 `app/agent/aiops/replanner.py` 中：

1. 在文件顶部 import 区新增：
```python
from app.observability import trace_node
```

2. 在 `async def replanner(state: PlanExecuteState) -> Dict[str, Any]:` 上方加装饰器：
```python
@trace_node("replanner")
async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
```

- [ ] **Step 5: 在 memory_writer.py 应用装饰器**

在 `app/agent/aiops/memory_writer.py` 中：

1. 在文件顶部 import 区（`from .state import PlanExecuteState` 之后）新增：
```python
from app.observability import trace_node
```

2. 在 `async def memory_writer(state: PlanExecuteState) -> Dict[str, Any]:` 上方加装饰器：
```python
@trace_node("memory_writer")
async def memory_writer(state: PlanExecuteState) -> Dict[str, Any]:
```

- [ ] **Step 6: 验证 4 个节点语法正确**

Run: `python -c "from app.agent.aiops import planner, executor, replanner, make_memory_writer; print('OK')"`
Expected: 输出 `OK`（无 ImportError / SyntaxError）

- [ ] **Step 7: 提交**

```bash
git add app/observability/instrumentation.py app/agent/aiops/planner.py app/agent/aiops/executor.py app/agent/aiops/replanner.py app/agent/aiops/memory_writer.py
git commit -m "feat(observability): @trace_node 装饰器应用到 4 个 AIOps 节点"
```

---

## Task 5: 工具调用 + Token 埋点（executor.py）

**Files:**
- Modify: `app/agent/aiops/executor.py`

**目标：** 在 executor 执行工具调用时，记录每个工具的名称/成功/失败/耗时/Token。Token 从 LLM 响应的 `usage_metadata` 提取。

- [ ] **Step 1: 在 executor.py 工具调用处添加埋点**

在 `app/agent/aiops/executor.py` 中，找到工具调用区块（约第 92-106 行），将以下代码：

```python
        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")
            
            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})
            
            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
```

替换为：

```python
        # 提取 LLM 响应的 Token 用量（如果模型返回了 usage_metadata）
        def _extract_tokens(resp) -> int:
            usage = getattr(resp, "usage_metadata", None)
            if usage and isinstance(usage, dict):
                return usage.get("total_tokens", 0)
            return 0

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            # 使用 ToolNode 自动执行工具（计时）
            messages.append(llm_response)
            import time
            _tool_start = time.time()
            tool_messages = await tool_node.ainvoke({"messages": messages})
            _tool_duration_ms = (time.time() - _tool_start) * 1000

            # ===== 工具调用埋点：记录每个工具的指标 =====
            from app.observability import metrics_collector, current_trace
            if current_trace() is not None:
                tool_result_map = {
                    getattr(m, "tool_call_id", ""): m
                    for m in tool_messages.get("messages", [])
                }
                for tc in llm_response.tool_calls:
                    _tool_name = tc.get("name", "unknown")
                    _tool_call_id = tc.get("id", "")
                    _result_msg = tool_result_map.get(_tool_call_id)
                    _tool_success = _result_msg is not None
                    _tool_error = ""
                    if not _tool_success:
                        _tool_error = "工具未返回结果"
                    elif "error" in str(getattr(_result_msg, "content", "")).lower():
                        _tool_success = False
                        _tool_error = "工具返回错误"
                    # 每个工具均摊耗时（ToolNode 批量执行，无法精确单工具计时）
                    _per_tool_duration = _tool_duration_ms / len(llm_response.tool_calls)
                    await metrics_collector.record_tool_call(
                        tool_name=_tool_name,
                        node_name="executor",
                        success=_tool_success,
                        duration_ms=round(_per_tool_duration, 2),
                        token_usage=0,  # 工具本身不消耗 LLM Token
                        error_message=_tool_error,
                    )
                    logger.info(
                        f"工具埋点: {_tool_name}, success={_tool_success}, "
                        f"duration={_per_tool_duration:.0f}ms"
                    )
            # ===== 埋点结束 =====

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)

            # 提取最终 LLM 响应的 Token 用量，累计到 trace
            _final_tokens = _extract_tokens(final_response)
            _first_tokens = _extract_tokens(llm_response)
            from app.observability import current_trace as _get_trace
            _trace_ctx = _get_trace()
            if _trace_ctx is not None and (_final_tokens + _first_tokens) > 0:
                _trace_ctx.total_tokens += _final_tokens + _first_tokens
                logger.debug(f"Token 用量: first_llm={_first_tokens}, final_llm={_final_tokens}")
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # 提取单次 LLM 调用的 Token 用量
            _tokens = _extract_tokens(llm_response)
            from app.observability import current_trace as _get_trace
            _trace_ctx = _get_trace()
            if _trace_ctx is not None and _tokens > 0:
                _trace_ctx.total_tokens += _tokens
                logger.debug(f"Token 用量: llm={_tokens}")
```

- [ ] **Step 2: 验证 executor.py 语法**

Run: `python -c "from app.agent.aiops.executor import executor; print('OK')"`
Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add app/agent/aiops/executor.py
git commit -m "feat(observability): executor 工具调用埋点 + Token 用量提取"
```

---

## Task 6: 服务层集成（aiops_service.py + main.py）

**Files:**
- Modify: `app/services/aiops_service.py`
- Modify: `app/main.py`

- [ ] **Step 1: 在 aiops_service.py 的 execute() 中集成 trace 生命周期**

在 `app/services/aiops_service.py` 中：

1. 在文件顶部 import 区新增（`from app.config import config` 之后）：
```python
from app.observability import start_trace, observability_store
```

2. 将 `execute()` 方法的 try 块（约第 154-219 行）中的核心执行部分包裹在 `start_trace` 中。

找到：
```python
        try:
            # 初始化持久化（首次调用时建立 AsyncSqliteSaver + 经验表，重建 graph）
            await self._initialize_persistence()

            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": ""
            }

            # 流式执行工作流
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates"
            ):
                # 解析事件
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    # 根据节点类型生成不同的事件
                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)

                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)

                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

                    elif node_name == NODE_MEMORY_WRITER:
                        yield self._format_memory_writer_event(node_output)

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""

            # 安全地获取响应（处理 values 可能为 None 的情况）
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }
```

替换为：

```python
        try:
            # 初始化持久化（首次调用时建立 AsyncSqliteSaver + 经验表，重建 graph）
            await self._initialize_persistence()

            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": ""
            }

            # 流式执行工作流
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # ===== 使用 start_trace 包裹整个执行过程 =====
            with start_trace(session_id=session_id, input_text=user_input) as trace_ctx:
                async for event in self.graph.astream(
                    input=initial_state,
                    config=config_dict,
                    stream_mode="updates"
                ):
                    # 解析事件
                    for node_name, node_output in event.items():
                        logger.info(f"节点 '{node_name}' 输出事件")

                        # 根据节点类型生成不同的事件
                        if node_name == NODE_PLANNER:
                            yield self._format_planner_event(node_output)

                        elif node_name == NODE_EXECUTOR:
                            yield self._format_executor_event(node_output)

                        elif node_name == NODE_REPLANNER:
                            yield self._format_replanner_event(node_output)

                        elif node_name == NODE_MEMORY_WRITER:
                            yield self._format_memory_writer_event(node_output)

                # 获取最终状态
                final_state = self.graph.get_state(config_dict)
                final_response = ""

                # 安全地获取响应（处理 values 可能为 None 的情况）
                if final_state and final_state.values:
                    final_response = final_state.values.get("response", "")

            # ===== trace 在 with 退出时自动结束 =====

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }
```

- [ ] **Step 2: 在 main.py lifespan 中初始化 observability store**

在 `app/main.py` 中：

1. 在 import 区（`from app.api import chat, health, file, aiops` 之后）新增：
```python
from app.observability import observability_store
```

2. 在 lifespan 的 yield 之前（Milvus 连接之后、Redis 检查之后），新增：
```python
    # 初始化可观测数据存储（Trace/Span/Metric）
    logger.info("📊 正在初始化可观测数据存储...")
    await observability_store.initialize()
    if observability_store.available:
        logger.info("✅ 可观测数据存储已就绪（Trace/Span/Metric）")
    else:
        logger.warning("⚠️ 可观测数据存储未就绪（埋点将降级为零开销直通）")
```

3. 在 lifespan 的 yield 之后（关闭 Milvus 之前），新增：
```python
    # 关闭可观测数据存储
    await observability_store.cleanup()
```

- [ ] **Step 3: 验证语法**

Run: `python -c "from app.services.aiops_service import aiops_service; from app.main import app; print('OK')"`
Expected: 输出 `OK`（可能伴随一些初始化日志，无 ImportError / SyntaxError 即可）

- [ ] **Step 4: 提交**

```bash
git add app/services/aiops_service.py app/main.py
git commit -m "feat(observability): execute() 集成 trace 生命周期 + lifespan 初始化 store"
```

---

## Task 7: 查询 API + 端到端验证

**Files:**
- Modify: `app/api/aiops.py`

- [ ] **Step 1: 在 api/aiops.py 新增查询端点**

在 `app/api/aiops.py` 的 import 区新增：
```python
from app.observability import observability_store, metrics_collector
```

在文件末尾（`diagnose_stream` 函数之后）新增两个端点：

```python
@router.get("/aiops/metrics")
async def get_metrics():
    """获取可观测聚合指标

    返回工具调用统计（成功率/延迟/Token）+ 节点执行统计 + 最近 trace 列表。
    用于运维仪表盘和面试演示。
    """
    try:
        summary = await metrics_collector.get_summary()
        return {"status": "success", "data": summary}
    except Exception as e:
        logger.error(f"获取指标失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.get("/aiops/traces/{trace_id}")
async def get_trace(trace_id: str):
    """查询单条 trace 的完整链路（含 spans + tool_metrics）

    用于全链路复现：输入什么 → planner 生成什么计划 → executor 调了哪些工具 →
    replanner 做了什么决策 → memory_writer 写了什么 → 最终响应。
    """
    try:
        trace = await observability_store.get_trace(trace_id)
        if trace is None:
            return {"status": "not_found", "message": f"trace_id {trace_id} 不存在"}
        return {"status": "success", "data": trace}
    except Exception as e:
        logger.error(f"查询 trace 失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.get("/aiops/traces")
async def list_traces(limit: int = 20):
    """列出最近的 trace（不含 spans 明细）"""
    try:
        traces = await observability_store.list_traces(limit=limit)
        return {"status": "success", "data": traces, "count": len(traces)}
    except Exception as e:
        logger.error(f"列出 trace 失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
```

- [ ] **Step 2: 验证 API 语法**

Run: `python -c "from app.api.aiops import router; print([r.path for r in router.routes])"`
Expected: 输出包含 `/aiops`、`/aiops/metrics`、`/aiops/traces/{trace_id}`、`/aiops/traces`

- [ ] **Step 3: 提交**

```bash
git add app/api/aiops.py
git commit -m "feat(observability): 新增 /metrics、/traces、/traces/{id} 查询端点"
```

- [ ] **Step 4: 端到端验证 - 启动应用**

Run: `python -m uvicorn app.main:app --host 127.0.0.1 --port 9900`
Expected: 应用启动，日志中应看到：
- `✅ 可观测数据存储已就绪（Trace/Span/Metric）`
- `INFO: Uvicorn running on http://127.0.0.1:9900`

- [ ] **Step 5: 端到端验证 - 触发 AIOps 执行**

在另一个终端执行（会触发完整 AIOps 流程）：
```bash
curl -X POST "http://127.0.0.1:9900/api/aiops" -H "Content-Type: application/json" -d "{\"session_id\": \"trace-test-1\"}" --no-buffer
```
Expected: 流式返回 SSE 事件，最终收到 `complete` 事件。应用日志中应看到：
- `[Trace xxxxxxxx] 开始追踪: ...`
- `[Span planner] 开始执行` / `[Span planner] 结束: status=completed, duration=...ms`
- `[Span executor] 开始执行` / 工具埋点日志
- `[Trace xxxxxxxx] 追踪结束: status=completed, duration=...ms, tokens=..., nodes=..., tools=...`

- [ ] **Step 6: 端到端验证 - 查询指标**

```bash
curl "http://127.0.0.1:9900/api/aiops/metrics" | python -m json.tool
```
Expected: 返回 JSON，包含：
- `global.global_tool_success_rate`（如 1.0）
- `tool_metrics` 数组（含 retrieve_knowledge / query_prometheus_alerts 等）
- `node_metrics` 数组（含 planner / executor / replanner / memory_writer）
- `recent_traces` 数组（含刚刚执行的 trace）

- [ ] **Step 7: 端到端验证 - 查询 trace 明细**

从 Step 6 的 `recent_traces` 中取一个 `trace_id`，执行：
```bash
curl "http://127.0.0.1:9900/api/aiops/traces/{trace_id}" | python -m json.tool
```
Expected: 返回完整 trace，含：
- `spans` 数组（每个节点的输入/输出摘要/耗时/状态）
- `tool_metrics` 数组（工具名/成功/耗时）
- 可用于全链路复现

- [ ] **Step 8: 端到端验证 - 降级测试**

在 `.env` 中设置 `OBSERVABILITY_ENABLED=false`（或在 config.py 中临时改为 False），重启应用，触发 AIOps：
Expected: 应用正常运行，日志中无 `[Trace ...]` / `[Span ...]` 输出，`/metrics` 返回空数据，功能不受影响。

---

## 自检清单

**1. 需求覆盖：**
- ✅ ①Trace 全链路追踪：`traces` 表 + `start_trace` 上下文管理器 + `/traces/{id}` 端点
- ✅ ②Span 单步骤切片：`spans` 表 + `@trace_node` 装饰器记录输入/输出/耗时/状态
- ✅ ③Metric 聚合统计：`tool_metrics` 表 + `MetricsCollector` + `/metrics` 端点（成功率/延迟/Token）
- ⏸ ④Log 结构化日志：本轮未做（现有 Loguru 文本日志保留）
- ⏸ ⑤Fact 最终产出物：本轮未做（trace 含 response 但未独立 Fact 表）
- ⏸ 漏斗式量化评估：本轮未做（需 Log/Fact 支撑）

**2. 降级安全：**
- ✅ `observability_enabled=False` 时，所有埋点零开销直通
- ✅ 无 trace 上下文时（如节点被独立调用），`@trace_node` 直接执行原函数
- ✅ SQLite 初始化失败时，`observability_store.available=False`，所有 save 操作静默跳过
- ✅ 不影响现有 AIOps 业务逻辑（装饰器只包裹，不修改内部代码）

**3. 数据隔离：**
- ✅ 可观测数据存于独立 SQLite（`data/observability.db`），与 checkpoint 库（`data/chat.db`）隔离
- ✅ contextvars 传播 trace_id，不污染 PlanExecuteState 业务状态

---

## 执行方式选择

计划已保存至 `docs/superpowers/plans/2026-08-10-observability-trace-span-metric.md`。两种执行方式：

**1. Subagent-Driven（推荐）** - 每个 Task 分派独立 subagent 执行，Task 间有 review 检查点，迭代快

**2. Inline Execution** - 在当前会话中按 Task 顺序执行，带检查点

**选哪种？**
