"""SubAgent 基类：定义子 Agent 的核心抽象

对应理论文档《SubAgent 子任务分发》：
- Agent 即 Tool（子 Agent 包装成 LangChain Tool）
- 定义式创建模式（预定义角色 + 空白上下文）
- 上下文隔离（独立 messages 数组、独立工具子集）
- RunToCompletion 执行模式（非交互式循环，直到 LLM 不再调工具）

设计要点：
- 工具过滤防线第 1 层 ALL_AGENT_DISALLOWED_TOOLS：
  子 Agent 不能再调用 sub_agent_tool，防止递归嵌套（A→B→C…）
- 共享基础设施（LLM 客户端、配置、信号量），隔离运行时状态
  （messages 数组、工具子集、Token 统计）
"""

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.services.llm_semaphore import get_llm_semaphore


# ============================================================
# 工具过滤防线：全局禁止列表
# ============================================================

# 第 1 层防线：所有子 Agent 都不能调用的工具名
# 防止子 Agent 递归调用 sub_agent_tool 造成嵌套爆炸（A→B→C→D…）
# 对应理论文档「工具过滤的多层防线」一节
ALL_AGENT_DISALLOWED_TOOLS = frozenset({
    "sub_agent_tool",  # 禁止子 Agent 再 spawn 子 Agent
})


class BaseSubAgent:
    """子 Agent 基类（定义式创建模式）

    子类通过类属性定义角色和能力：
    - agent_type: 唯一标识（如 "knowledge_agent"）
    - when_to_use: 用途描述（给主 Agent 看，决定何时调用）
    - tools: 工具白名单（子 Agent 可用的工具列表）
    - system_prompt: 角色 System Prompt
    - model: 使用的 LLM 模型名
    - max_turns: 最大循环次数（防止无限循环）

    执行入口：run_to_completion(task) -> str

    设计原则（对应理论文档）：
    - 上下文隔离：每次执行使用独立的 messages 数组，不污染主 Agent
    - RunToCompletion：非交互式循环，LLM 不再调工具时结束
    - 工具过滤：初始化时应用 ALL_AGENT_DISALLOWED_TOOLS 过滤
    - 失败容错：异常时不抛出，返回错误字符串，不阻塞主流程
    """

    # ====== 子类必须覆写的类属性 ======
    agent_type: str = ""           # 子 Agent 类型标识
    when_to_use: str = ""          # 用途描述（给主 Agent 决策参考）
    tools: List[BaseTool] = []     # 工具白名单
    system_prompt: str = ""        # 角色 System Prompt
    model: str = ""                # LLM 模型名（不指定则用 config.rag_model）
    max_turns: int = 5             # 最大循环次数

    def __init__(self):
        """初始化子 Agent

        应用第 1 层工具过滤防线：从 tools 中剔除全局禁止的工具
        （防止子 Agent 调用 sub_agent_tool 造成递归嵌套）
        """
        # 第 1 层过滤：剔除全局禁止的工具
        self._filtered_tools: List[BaseTool] = [
            t for t in self.tools
            if t.name not in ALL_AGENT_DISALLOWED_TOOLS
        ]

        # 校验子类配置完整性
        if not self.agent_type:
            raise ValueError(
                f"{self.__class__.__name__} 必须定义 agent_type 类属性"
            )
        if not self.system_prompt:
            raise ValueError(
                f"{self.__class__.__name__} 必须定义 system_prompt 类属性"
            )
        if not self.model:
            self.model = config.rag_model  # 默认使用 RAG 模型

    async def run_to_completion(self, task: str) -> str:
        """非交互式执行入口：循环调用 LLM+工具，直到完成

        对应理论文档 RunToCompletion：
        1. 任务直接从参数注入（不等用户输入）
        2. 循环：LLM 推理 → 工具调用 → 直到 LLM 不再调工具
        3. 返回最后一次有文本内容的 LLM 响应

        执行流程示例（KnowledgeAgent 检索知识）：
        - 第 1 轮：LLM 决定调用 retrieve_knowledge → 工具执行 → 工具结果加入 messages
        - 第 2 轮：LLM 看到工具结果，生成最终摘要文本（无 tool_calls）→ 循环结束
        - 返回最终文本

        Args:
            task: 子任务描述（主 Agent 传入的 prompt）

        Returns:
            str: 子 Agent 的最终输出文本。失败时返回错误说明字符串（不抛异常）
        """
        logger.info(f"=== SubAgent[{self.agent_type}]: run_to_completion 开始 ===")
        logger.info(
            f"任务: {task[:100]}{'...' if len(task) > 100 else ''}"
        )

        # 空任务保护
        if not task or not task.strip():
            logger.warning("任务为空，子 Agent 直接返回空字符串")
            return ""

        try:
            # 构建 LLM 客户端（共享 LLM 配置，但实例独立）
            llm = ChatQwen(
                model=self.model,
                api_key=config.dashscope_api_key,
                base_url=config.dashscope_api_base,
                temperature=0
            )
            llm_with_tools = llm.bind_tools(self._filtered_tools)

            # 工具节点（自动执行工具调用，处理 ToolMessage 流）
            tool_node = ToolNode(self._filtered_tools)

            # ====== 上下文隔离的关键：独立的 messages 数组 ======
            # 与主 Agent 的对话历史完全隔离
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=task),
            ]

            last_text = ""

            # RunToCompletion 主循环
            for turn in range(self.max_turns):
                logger.info(
                    f"SubAgent[{self.agent_type}] 第 {turn + 1}/{self.max_turns} 轮"
                )

                # 第一步：LLM 推理（受全局并发信号量控制）
                async with get_llm_semaphore():
                    llm_response = await llm_with_tools.ainvoke(messages)
                messages.append(llm_response)

                # 提取文本（LLM 在调用工具时可能不返回文本，纯工具调用）
                if hasattr(llm_response, 'content') and llm_response.content:
                    last_text = llm_response.content

                # 第二步：检查是否需要调用工具
                tool_calls = getattr(llm_response, 'tool_calls', None)
                if not tool_calls:
                    # LLM 不调工具 = 任务完成
                    logger.info(
                        f"SubAgent[{self.agent_type}] 第 {turn + 1} 轮无工具调用，任务完成"
                    )
                    break

                # 第三步：执行工具调用
                logger.info(
                    f"SubAgent[{self.agent_type}] 调用 {len(tool_calls)} 个工具: "
                    f"{[tc.get('name', 'unknown') for tc in tool_calls]}"
                )
                tool_messages = await tool_node.ainvoke({"messages": messages})
                messages.extend(tool_messages["messages"])
            else:
                # for...else：循环走完所有迭代没 break（达到 max_turns）
                logger.warning(
                    f"SubAgent[{self.agent_type}] 达到最大循环次数 "
                    f"{self.max_turns}，强制结束"
                )
                if not last_text:
                    last_text = (
                        f"子 Agent 达到最大循环次数 {self.max_turns}，"
                        f"未能生成最终响应"
                    )

            logger.info(
                f"SubAgent[{self.agent_type}] 执行完成，结果长度: {len(last_text)}"
            )
            return last_text

        except Exception as e:
            # 失败容错：不抛异常，返回错误字符串，避免阻塞主流程
            logger.error(
                f"SubAgent[{self.agent_type}] 执行失败: {e}",
                exc_info=True
            )
            return f"子 Agent 执行失败: {str(e)}"
