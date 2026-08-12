"""工具模块 - 供 Agent 调用的各种工具"""

from app.tools.knowledge_tool import retrieve_knowledge
from app.tools.query_metrics_alerts import query_prometheus_alerts
from app.tools.time_tool import get_current_time

# 导入 sub_agent_tool 时会触发 app.agent.sub_agents 模块加载，
# 自动在 AgentRegistry 中注册 KnowledgeAgent
from app.agent.sub_agents import sub_agent_tool

# 默认本地工具集：凡绑定「知识库 + 时间」的 Agent 应使用此元组，
# 与 Prometheus 告警查询一并注册；sub_agent_tool 让主 Agent 可调用子 Agent
DEFAULT_LOCAL_AGENT_TOOLS = (
    retrieve_knowledge,
    get_current_time,
    query_prometheus_alerts,
    sub_agent_tool,
)

__all__ = [
    "DEFAULT_LOCAL_AGENT_TOOLS",
    "retrieve_knowledge",
    "get_current_time",
    "query_prometheus_alerts",
    "sub_agent_tool",
]
