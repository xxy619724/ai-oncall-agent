"""可观测数据 SQLite 存储层

三张表：
- traces：一次 AIOps 执行的全链路记录
- spans：单节点执行切片
- tool_metrics：工具调用指标（用于聚合统计成功率/延迟）

所有方法均为 async，使用 aiosqlite 连接。
连接由 main.py lifespan 初始化，全局单例 observability_store。
"""

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
