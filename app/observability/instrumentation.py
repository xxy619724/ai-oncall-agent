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
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from loguru import logger

from app.config import config
from app.observability.metrics import metrics_collector
from app.observability.store import observability_store
from app.observability.trace import current_trace, schedule_write, _state_to_summary


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

                # 异步持久化（旁路落库，不阻塞主流程）
                # 用 schedule_write 而非裸 ensure_future：后者不持强引用，
                # 协程可能在执行前被 GC，且内部异常无人接收
                schedule_write(observability_store.save_span(span_data))
                schedule_write(
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
