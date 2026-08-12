"""SubAgent 模块：多 Agent 协作的核心

对应理论文档《SubAgent 子任务分发》：
- Agent 即 Tool（子 Agent 包装成 LangChain Tool）
- 定义式创建模式（Python 类形式）
- 上下文隔离（独立 messages + 工具子集）
- 工具过滤多层防线（防递归嵌套）

模块结构：
- base.py: BaseSubAgent 基类 + RunToCompletion 循环
- registry.py: AgentRegistry 单例（注册/查找子 Agent）
- knowledge_agent.py: 第一个子 Agent 实现（知识库检索）
- agent_tool.py: sub_agent_tool LangChain Tool 包装

使用方式：

    # 方式 1：在代码中显式调用（planner 使用）
    from app.agent.sub_agents import sub_agent_tool

    result = await sub_agent_tool.ainvoke({
        "subagent_type": "knowledge_agent",
        "prompt": "检索与XX相关的知识",
        "description": "为规划阶段提供经验支撑"
    })

    # 方式 2：作为 LLM 工具自动调用（executor 使用）
    from app.agent.sub_agents import sub_agent_tool

    llm_with_tools = llm.bind_tools([sub_agent_tool, ...])
    # LLM 在推理时自行决定是否调用 sub_agent_tool

扩展方式（添加新的子 Agent）：
    1. 继承 BaseSubAgent，定义 agent_type / tools / system_prompt 等类属性
    2. 在本 __init__.py 末尾添加 AgentRegistry.register(YourAgent())
"""

from .base import ALL_AGENT_DISALLOWED_TOOLS, BaseSubAgent
from .registry import AgentRegistry
from .knowledge_agent import KnowledgeAgent
from .agent_tool import sub_agent_tool

# ====== 模块加载时自动注册内置子 Agent ======
# 这样外部模块只需 import sub_agent_tool 即可使用，
# 不需要手动注册 KnowledgeAgent
# AgentRegistry.register 内部有覆盖保护，重复导入不会出问题
AgentRegistry.register(KnowledgeAgent())


__all__ = [
    "BaseSubAgent",
    "AgentRegistry",
    "KnowledgeAgent",
    "sub_agent_tool",
    "ALL_AGENT_DISALLOWED_TOOLS",
]
