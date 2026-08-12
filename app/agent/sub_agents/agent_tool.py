"""AgentTool：把子 Agent 包装成 LangChain Tool

对应理论文档核心思想「Agent 即 Tool」：
- 统一一个 Agent 工具，通过 subagent_type 参数选择不同子 Agent
- 主 Agent 调用它的方式跟调用其他工具一模一样

为什么是统一工具而不是每个子 Agent 一个独立工具？
对应理论文档原文：
> 因为 Agent 类型可以动态加载。如果每个类型都注册一个独立工具，
> 工具列表会随着定义文件的增减而变化，系统提示也要跟着重新渲染。
> 统一成一个 Agent 工具，通过 subagent_type 参数选择类型，工具列表始终稳定。

被调用方式（两种）：

1. planner.py 代码中显式调用：
    result = await sub_agent_tool.ainvoke({
        "subagent_type": "knowledge_agent",
        "prompt": "检索与XX相关的知识",
        "description": "为规划阶段提供经验支撑"
    })

2. executor.py 通过 LLM 自动决定：
    llm_with_tools = llm.bind_tools([sub_agent_tool, ...])
    # LLM 在推理时自行决定是否调用 sub_agent_tool

可观测性埋点（P1 修复）：
- 通过 current_trace() 读取 contextvar 中的 trace 上下文
- 无 trace 上下文时零开销降级（直接执行，不埋点）
- 有 trace 上下文时，记录 span_type="sub_agent" 的 span
  - node_name 用 subagent_type（如 "knowledge_agent"），便于按子 Agent 类型聚合
  - 异步保存到 observability_store，不阻塞主流程
- contextvars 在 async 链路中自动传播，子 Agent 调用能读到 planner/executor 设置的 trace_id
"""

import asyncio
import json
import uuid
from datetime import datetime

from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.observability import current_trace, metrics_collector, observability_store
from .registry import AgentRegistry


def _truncate_for_span(text: str, max_len: int = 500) -> str:
    """截断文本到指定长度，超长加省略号（用于 span 摘要）"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"


@tool
async def sub_agent_tool(
    subagent_type: str,
    prompt: str,
    description: str = "",
) -> str:
    """调用专门的子 Agent 完成子任务（在独立上下文中执行）

    当某个子任务需要专门角色（如知识检索、代码探索、计划分析）时使用此工具。
    子 Agent 会在独立上下文中执行（不污染主 Agent 对话），完成后返回结果。

    Args:
        subagent_type: 子 Agent 类型，目前支持:
            - "knowledge_agent": 知识库检索专家，从内部知识库查文档/经验/最佳实践
        prompt: 子任务描述，要清晰具体，子 Agent 会作为指令执行
        description: 任务用途说明（便于观测，可选）

    Returns:
        str: 子 Agent 的执行结果文本。失败时返回错误说明字符串
    """
    logger.info(
        f"AgentTool 调用: type='{subagent_type}', "
        f"description='{description or 'N/A'}', "
        f"prompt='{prompt[:80]}{'...' if len(prompt) > 80 else ''}'"
    )

    # ============================================================
    # 可观测性：span 开始（无 trace 上下文时跳过埋点）
    # ============================================================
    trace_ctx = current_trace()
    span_id = str(uuid.uuid4()) if trace_ctx is not None else ""
    span_started_at = datetime.now().isoformat() if trace_ctx is not None else ""

    if trace_ctx is not None:
        # 构建输入摘要（JSON 格式，便于查询）
        input_summary = json.dumps({
            "subagent_type": subagent_type,
            "description": _truncate_for_span(description, 200),
            "prompt": _truncate_for_span(prompt, 300),
        }, ensure_ascii=False)
        logger.debug(f"[Span sub_agent:{subagent_type}] 开始执行")
    else:
        input_summary = ""

    # ============================================================
    # 执行子 Agent
    # ============================================================
    status = "completed"
    output_summary: str = ""
    error_message = ""

    # 先从注册表查找子 Agent
    agent = AgentRegistry.resolve(subagent_type)
    if agent is None:
        available = list(AgentRegistry.list_agents().keys())
        error_msg = (
            f"错误：未找到子 Agent 类型 '{subagent_type}'。"
            f"当前可用类型: {available}"
        )
        logger.warning(error_msg)
        # 记录失败 span（如果有 trace 上下文）
        if trace_ctx is not None:
            status = "failed"
            error_message = error_msg
            output_summary = _truncate_for_span(error_msg, 500)
            await _finalize_span(
                trace_ctx, span_id, span_started_at, subagent_type,
                input_summary, output_summary, status, error_message,
            )
        return error_msg

    # 调用子 Agent 执行任务（RunToCompletion 模式）
    # 子 Agent 内部有失败容错，正常情况下不会抛异常
    try:
        result = await agent.run_to_completion(prompt)
        output_summary = _truncate_for_span(result, 500)

        logger.info(
            f"AgentTool 完成: type='{subagent_type}', 结果长度={len(result)}"
        )
        return result

    except Exception as e:
        # 兜底：即使子 Agent 容错失败，也保证 span 被记录
        status = "failed"
        error_message = f"{type(e).__name__}: {e}"
        output_summary = _truncate_for_span(error_message, 500)
        logger.error(f"AgentTool 异常: type='{subagent_type}', error={e}", exc_info=True)
        raise

    finally:
        # ============================================================
        # 可观测性：span 结束（异步保存，不阻塞主流程）
        # ============================================================
        if trace_ctx is not None:
            await _finalize_span(
                trace_ctx, span_id, span_started_at, subagent_type,
                input_summary, output_summary, status, error_message,
            )


async def _finalize_span(
    trace_ctx,
    span_id: str,
    started_at: str,
    subagent_type: str,
    input_summary: str,
    output_summary: str,
    status: str,
    error_message: str,
) -> None:
    """收尾 span：计算耗时 + 异步保存到 store + 记录指标

    将 span 保存逻辑抽离为独立函数，便于在正常返回和异常路径统一调用。
    所有持久化操作 fire-and-forget，不阻塞主流程。
    """
    ended_at = datetime.now().isoformat()
    start_dt = datetime.fromisoformat(started_at)
    duration_ms = round((datetime.now() - start_dt).total_seconds() * 1000, 2)

    # 构建 metadata（含错误信息，便于排查）
    span_metadata = json.dumps(
        {"error": error_message} if error_message else {},
        ensure_ascii=False,
    )

    span_data = {
        "span_id": span_id,
        "trace_id": trace_ctx.trace_id,
        "node_name": subagent_type,  # 用子 Agent 类型作为 node_name，便于聚合
        "span_type": "sub_agent",   # 区分普通 node span
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "token_usage": 0,  # 子 Agent 内部 Token 由其自身埋点统计（未来扩展）
        "metadata": span_metadata,
    }

    # 异步保存 span（fire-and-forget）
    asyncio.ensure_future(observability_store.save_span(span_data))

    # 异步记录节点指标（累计到 trace 上下文）
    asyncio.ensure_future(
        metrics_collector.record_node_completion(
            node_name=subagent_type,
            duration_ms=duration_ms,
            token_usage=0,
            status=status,
        )
    )

    logger.debug(
        f"[Span sub_agent:{subagent_type}] 结束: status={status}, "
        f"duration={duration_ms}ms"
    )
