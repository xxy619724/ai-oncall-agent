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
