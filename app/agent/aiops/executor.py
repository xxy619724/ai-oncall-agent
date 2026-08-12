"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS
from app.agent.mcp_client import get_mcp_client_with_retry
from app.services.llm_semaphore import get_llm_semaphore
from .state import PlanExecuteState
from app.observability import trace_node


@trace_node("executor")
async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤
    
    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    # 目标锚定：取原始任务目标，确保执行单步时始终知道整体目标，避免目标漂移
    input_text = state.get("input", "")
    logger.info(f"当前任务: {task}")

    try:
        # 获取本地工具
        local_tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # 获取 MCP 工具（失败时降级为仅本地工具）
        try:
            mcp_client = await get_mcp_client_with_retry()
            mcp_tools = await mcp_client.get_tools()
        except Exception as e:
            logger.warning(f"MCP 工具获取失败，仅使用本地工具: {e}")
            mcp_tools = []
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建 LLM（绑定工具）
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 构建消息（含目标锚定区：注入原始任务目标，避免执行单步时目标漂移）
        messages = [
            SystemMessage(content=f"""你是一个能力强大的助手，负责执行具体的任务步骤。

【目标锚定】原始任务目标: {input_text}

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        # 第一步：LLM 决定是否调用工具（受 LLM 并发信号量控制）
        async with get_llm_semaphore():
            llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

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

            # 第三步：将工具结果返回给 LLM 生成最终答案（受 LLM 并发信号量控制）
            messages.extend(tool_messages["messages"])
            async with get_llm_semaphore():
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

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，添加执行历史
        return {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }
