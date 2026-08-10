"""MetricsCollector 单例：运行期指标记录 + 聚合查询

记录维度：
- 节点级：执行次数/成功失败/耗时/Token
- 工具级：调用次数/成功率/延迟/Token

数据流：
  节点/工具执行 → MetricsCollector.record_xxx() → store.save_xxx() → SQLite
  查询时 → store.get_xxx_summary() → 聚合统计

无 trace 上下文时，record 方法自动跳过（零开销降级）。
"""

from typing import Any, Dict

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
