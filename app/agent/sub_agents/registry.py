"""Agent Registry：子 Agent 注册表

对应理论文档「Agent 即 Tool」设计：
- 统一一个 sub_agent_tool 工具，通过 subagent_type 参数选择不同子 Agent
- 注册表负责管理和查找所有可用的子 Agent 实例

设计原则：
- 单例模式：类变量作为全局注册表
- 延迟注册：在 sub_agents/__init__.py 中显式注册内置 Agent
- 简单查找：通过 agent_type 字符串查找子 Agent 实例

用法示例：

    # 注册子 Agent（通常在 __init__.py 中完成）
    AgentRegistry.register(KnowledgeAgent())

    # 查找子 Agent
    agent = AgentRegistry.resolve("knowledge_agent")
    result = await agent.run_to_completion("检索与XX相关的知识")
"""

from typing import Dict, Optional

from loguru import logger

from .base import BaseSubAgent


class AgentRegistry:
    """子 Agent 注册表（单例）

    使用类变量作为全局存储，整个进程共享一份注册表。
    所有方法都是 classmethod，不需要实例化即可使用。
    """

    # 全局注册表：agent_type -> BaseSubAgent 实例
    _agents: Dict[str, BaseSubAgent] = {}

    @classmethod
    def register(cls, agent: BaseSubAgent) -> None:
        """注册子 Agent

        如果 agent_type 已存在，会覆盖旧注册（并打印警告）。

        Args:
            agent: BaseSubAgent 子类实例
        """
        if not isinstance(agent, BaseSubAgent):
            raise TypeError(
                f"agent 必须是 BaseSubAgent 实例，got {type(agent).__name__}"
            )

        agent_type = agent.agent_type
        if agent_type in cls._agents:
            logger.warning(
                f"子 Agent '{agent_type}' 已注册，将被覆盖（旧: "
                f"{cls._agents[agent_type].__class__.__name__}, 新: "
                f"{agent.__class__.__name__}）"
            )

        cls._agents[agent_type] = agent
        logger.info(
            f"子 Agent 已注册: type='{agent_type}', "
            f"class={agent.__class__.__name__}, "
            f"tools={[t.name for t in agent.tools]}"
        )

    @classmethod
    def resolve(cls, agent_type: str) -> Optional[BaseSubAgent]:
        """查找子 Agent

        Args:
            agent_type: 子 Agent 类型标识（如 "knowledge_agent"）

        Returns:
            BaseSubAgent 实例；找不到返回 None
        """
        agent = cls._agents.get(agent_type)
        if agent is None:
            available = list(cls._agents.keys())
            logger.warning(
                f"子 Agent '{agent_type}' 未注册。当前可用: {available}"
            )
        return agent

    @classmethod
    def list_agents(cls) -> Dict[str, BaseSubAgent]:
        """列出所有已注册的子 Agent（用于调试和文档生成）"""
        return dict(cls._agents)

    @classmethod
    def clear(cls) -> None:
        """清空注册表

        仅用于单元测试隔离，生产环境不应调用。
        """
        cls._agents.clear()
        logger.info("AgentRegistry 已清空")
