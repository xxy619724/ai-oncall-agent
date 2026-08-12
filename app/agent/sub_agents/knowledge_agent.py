"""KnowledgeAgent：知识库检索子 Agent

第一个落地的子 Agent 实现，对应理论文档「定义式」创建模式：
- 角色：知识检索专家
- 能力：只能调用 retrieve_knowledge 工具
- 上下文：独立 messages 数组（不污染主 Agent）
- 工具过滤：经 ALL_AGENT_DISALLOWED_TOOLS 过滤后已无 sub_agent_tool

被调用方式（两种统一为同一种 AgentTool 调用）：
1. planner.py 代码显式 await sub_agent_tool.ainvoke({...})
   （替代原来直接 ainvoke retrieve_knowledge 的方式）
2. executor.py 通过 LLM 自动决定是否调用 sub_agent_tool
   （工具集自动包含，LLM 自主选择）

设计意图：
- 给"知识检索"赋予专门角色和提示词，提升检索质量
- 统一 planner 和 executor 的调用方式（都通过 AgentTool）
- 为未来添加更多子 Agent（ExploreAgent、PlanAgent）建立模式

注意（循环导入处理）：
不在模块顶层导入 retrieve_knowledge，而是在 __init__ 中延迟导入。
原因：app.tools.__init__ 顶部导入 sub_agent_tool → 触发 sub_agents 包加载
→ knowledge_agent 顶层导入 retrieve_knowledge → 触发 app.tools 包加载（循环）。
延迟到 __init__ 时 app.tools 已加载完成，可安全导入。
"""

from typing import List

from langchain_core.tools import BaseTool

from app.config import config

from .base import BaseSubAgent


class KnowledgeAgent(BaseSubAgent):
    """知识库检索子 Agent

    职责：从向量库（Milvus）检索相关知识 + Rerank 重排 + 格式化返回

    工具：[retrieve_knowledge]
        retrieve_knowledge 内部已封装：
        - 向量检索（top_k=config.rag_top_k，COSINE 相似度）
        - Rerank 重排（top_k=config.rag_rerank_top_k）
        - 格式化输出（含来源、标题、相关性分数）

    适用场景：
    - planner 在制定计划前查历史经验
    - executor 在执行步骤时需要专业知识参考
    - 用户问专业问题时检索内部文档
    """

    # ====== 类属性：定义子 Agent 的角色和能力 ======

    agent_type = "knowledge_agent"

    when_to_use = (
        "当需要从内部知识库检索文档、历史经验、最佳实践、专业知识时使用。"
        "适用场景：规划任务前查经验、执行步骤时需要参考资料、"
        "回答专业问题、查找运维经验。"
    )

    # 工具白名单：留空，在 __init__ 中通过延迟导入赋值
    # 不在类属性直接赋值是为了避免模块加载时的循环导入
    # （详见模块顶部注释）
    tools: List[BaseTool] = []

    # 角色 System Prompt：定义子 Agent 的身份和行为约束
    # 对应理论文档「Agent 定义也是 Markdown」中的 body 部分
    system_prompt = """你是一个知识检索专家。你的唯一任务是从内部知识库检索相关信息。

## 职责
- 根据用户的查询关键词，调用 retrieve_knowledge 工具检索知识
- 把检索到的资料整理成清晰的文本返回给调用方

## 规则（不可协商）
1. 只使用 retrieve_knowledge 工具，不要尝试调用其他工具
2. 不闲聊，不回答与知识检索无关的问题
3. 查询参数用简洁的关键词，不要传整段话（影响检索质量）
4. 输出限制在 800 字以内，避免冗余

## 输出格式
直接返回检索到的知识内容，可以适当添加简短的总结说明。
不要添加"我检索到了..."这类元描述，直接给结果。
如果未检索到相关内容，明确说明"未找到相关资料"。
"""

    # 模型：使用 RAG 模型（与 retrieve_knowledge 配套）
    model: str = config.rag_model

    # 最大循环次数：知识检索通常 2 轮即可（1 轮决策+工具调用，1 轮总结）
    # 设为 3 留一次余量，防止 LLM 需要重试检索
    max_turns: int = 3

    def __init__(self):
        """初始化 KnowledgeAgent

        通过延迟导入获取 retrieve_knowledge 工具，避免模块加载时的循环依赖。
        此时 app.tools 已加载完成，retrieve_knowledge 可安全导入。
        """
        # 延迟导入：打破 app.tools → sub_agents → knowledge_agent → app.tools 的循环
        from app.tools.knowledge_tool import retrieve_knowledge

        # 工具白名单：只能用知识检索工具
        # 注：BaseSubAgent.__init__ 会自动从中剔除 sub_agent_tool（防递归）
        self.tools = [retrieve_knowledge]

        # 调用父类初始化（会执行工具过滤）
        super().__init__()
