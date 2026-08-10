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
    "metrics_collector",
    "MetricsCollector",
    "trace_node",
]
