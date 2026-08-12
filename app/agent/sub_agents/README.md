# SubAgent 子任务分发：多 Agent 协作说明文档

> 本文档面向零基础读者，一步步讲清楚多 Agent 协作的原理、设计和用法。

---

## 目录

- [1. 一句话总览](#1-一句话总览)
- [2. 为什么需要多 Agent 协作](#2-为什么需要多-agent-协作)
- [3. 核心思想：Agent 即 Tool](#3-核心思想agent-即-tool)
- [4. 架构设计](#4-架构设计)
- [5. 文件结构与职责](#5-文件结构与职责)
- [6. 使用方式](#6-使用方式)
- [7. 工具过滤防线（防递归嵌套）](#7-工具过滤防线防递归嵌套)
- [8. 如何添加新的子 Agent（扩展指南）](#8-如何添加新的子-agent扩展指南)
- [9. 常见问题排查](#9-常见问题排查)
- [10. 与理论文档的映射关系](#10-与理论文档的映射关系)
- [11. P1 修复：可观测性集成（sub_agent_tool Span 埋点）](#11-p1-修复可观测性集成sub_agent_tool-span-埋点)

---

## 1. 一句话总览

本项目通过 **「Agent 即 Tool」** 的设计，把"知识库检索"包装成一个独立的子 Agent（KnowledgeAgent），让 planner 和 executor 通过统一的 `sub_agent_tool` 调用它，实现上下文隔离、角色专门化、防止主流程污染。

---

## 2. 为什么需要多 Agent 协作

### 2.1 单 Agent 的痛点：上下文污染

改造前，项目只有一个 Plan-Execute-Replan 工作流，所有任务在同一个对话上下文里执行：

```
planner（制定计划） → executor（执行步骤） → replanner（决策下一步） → 循环
```

问题来了：

- **planner 制定计划时**：直接调用 `retrieve_knowledge.ainvoke()` 检索知识，结果（可能几千 token）直接塞进 planner 的对话历史
- **executor 执行步骤时**：把 `retrieve_knowledge` 作为工具绑定给 LLM，LLM 检索后的中间过程（查询词、检索结果、总结）全部留在 executor 的对话历史
- **后果**：planner 和 executor 的上下文都被知识检索的中间过程"污染"了，后续步骤要在噪声中翻找有用信息，token 消耗飙升，响应质量下降

### 2.2 解决思路：子任务隔离

把"知识检索"这个子任务交给一个**专门的子 Agent**：

- 它有自己**独立的对话上下文**（messages 数组）
- 它有**专门的角色和提示词**（知识检索专家）
- 它有**受限的工具集**（只能用 retrieve_knowledge）
- 完成后**只把最终结果**返回给主 Agent，中间过程不外泄

这就是多 Agent 协作的核心价值。

### 2.3 一个生活化的比喻

想象你是一个公司老板（主 Agent），原本所有事都自己干：既规划项目、又查资料、又执行任务。问题是你查完资料的脑子还沉浸在细节里，立刻又要切换成"决策者"角色，转不过来。

**解决**：雇一个资料员（子 Agent），他有自己的办公室（独立上下文），你把任务给他，他在自己的小天地里查完资料，把结论汇报给你。你只看结论，不参与他的查阅过程。

---

## 3. 核心思想：Agent 即 Tool

### 3.1 关键洞察

回顾 Tool 的接口：
- 有名字
- 有描述
- 接受参数
- 返回结果

再看 Agent：
- 有名字（角色）
- 有描述（用途）
- 接受任务（参数）
- 返回结果

**两者的抽象完全同构**。所以可以把"子 Agent"包装成"工具"，主 Agent 调用它的方式跟调用其他工具一模一样。

### 3.2 统一 AgentTool 设计

我们设计了一个统一的 LangChain Tool：`sub_agent_tool`。它通过 `subagent_type` 参数选择不同的子 Agent：

```python
@tool
async def sub_agent_tool(
    subagent_type: str,    # 选哪个子 Agent（如 "knowledge_agent"）
    prompt: str,           # 任务描述
    description: str = "", # 任务用途说明（便于观测）
) -> str:
    """调用专门的子 Agent 完成子任务"""
    agent = AgentRegistry.resolve(subagent_type)
    return await agent.run_to_completion(prompt)
```

**为什么不每个子 Agent 注册一个独立工具？**

对应理论文档原文：
> 因为 Agent 类型可以动态加载。如果每个类型都注册一个独立工具，工具列表会随着定义文件的增减而变化，系统提示也要跟着重新渲染。统一成一个 Agent 工具，通过 `subagent_type` 参数选择类型，工具列表始终稳定。

### 3.3 两种调用方式

`sub_agent_tool` 作为一个 LangChain Tool，有两种使用方式：

**方式 1：代码中显式调用**（planner 使用）

```python
result = await sub_agent_tool.ainvoke({
    "subagent_type": "knowledge_agent",
    "prompt": "检索与XX相关的知识",
    "description": "为规划阶段提供经验支撑"
})
```

**方式 2：LLM 自动决定调用**（executor 使用）

```python
llm_with_tools = llm.bind_tools([sub_agent_tool, get_current_time, ...])
# LLM 在推理时自行决定是否调用 sub_agent_tool
```

两种方式走的是**完全相同**的执行路径，这就是"统一"的含义。

---

## 4. 架构设计

### 4.1 改造前的调用关系

```
planner ──(直接 ainvoke)──→ retrieve_knowledge
                                    ↑
executor ──(bind_tools)─────────────┘
```

- planner 直接调用 `retrieve_knowledge.ainvoke()`
- executor 把 `retrieve_knowledge` 绑定给 LLM
- **两种调用方式不统一**，且都没有 Agent 包装

### 4.2 改造后的调用关系

```
planner ──(显式 ainvoke)──→ sub_agent_tool ──→ KnowledgeAgent.run_to_completion()
                                                          │
                                                          ├── 独立 messages 数组
                                                          ├── SystemPrompt（知识检索专家）
                                                          └── 调用 retrieve_knowledge 工具
                                                                    ↑
executor ──(bind_tools)──→ sub_agent_tool ─────────────────────────┘
                            （LLM 自主决定是否调用）
```

- planner 和 executor 都通过 `sub_agent_tool` 调用 KnowledgeAgent
- KnowledgeAgent 在独立上下文中执行检索
- 中间过程不污染主流程

### 4.3 整体工作流图

```
┌─────────────────────────────────────────────────────────────────┐
│                    主 Agent 工作流（LangGraph）                  │
│                                                                 │
│  planner ──→ executor ──→ replanner ──→ memory_writer ──→ END  │
│                ↑                  │                             │
│                │                  ↓                             │
│                └────── (循环) ────┘                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  planner 和 executor 都可以通过 sub_agent_tool 调用子 Agent  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    子 Agent 执行环境（隔离）                     │
│                                                                 │
│  KnowledgeAgent.run_to_completion(task)                         │
│  ├── 独立 messages 数组（不污染主 Agent）                        │
│  ├── 独立 SystemPrompt（知识检索专家角色）                      │
│  ├── 工具子集（只含 retrieve_knowledge，过滤掉 sub_agent_tool）  │
│  └── RunToCompletion 循环（LLM + 工具，直到完成）               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 文件结构与职责

```
app/agent/sub_agents/
├── __init__.py            # 模块入口，导出 + 自动注册 KnowledgeAgent
├── base.py                # BaseSubAgent 基类 + RunToCompletion 循环
├── registry.py            # AgentRegistry 单例（注册/查找子 Agent）
├── knowledge_agent.py     # 第一个子 Agent：知识库检索专家
├── agent_tool.py          # sub_agent_tool LangChain Tool 包装
└── README.md              # 本说明文档
```

### 5.1 `base.py`：BaseSubAgent 基类

**职责**：定义子 Agent 的核心抽象，提供 `run_to_completion` 执行入口。

**关键设计**：
- 子类通过类属性定义角色：`agent_type` / `when_to_use` / `tools` / `system_prompt` / `model` / `max_turns`
- `__init__` 时应用第 1 层工具过滤（剔除 `sub_agent_tool`）
- `run_to_completion(task)` 是非交互式执行循环，直到 LLM 不再调工具为止

**核心代码片段**：

```python
class BaseSubAgent:
    agent_type: str = ""
    when_to_use: str = ""
    tools: List[BaseTool] = []
    system_prompt: str = ""
    model: str = ""
    max_turns: int = 5

    def __init__(self):
        # 第 1 层过滤：剔除全局禁止的工具
        self._filtered_tools = [
            t for t in self.tools
            if t.name not in ALL_AGENT_DISALLOWED_TOOLS
        ]

    async def run_to_completion(self, task: str) -> str:
        # 独立 messages 数组（上下文隔离的关键）
        messages = [SystemMessage(...), HumanMessage(content=task)]

        for turn in range(self.max_turns):
            # LLM 推理
            llm_response = await llm_with_tools.ainvoke(messages)
            messages.append(llm_response)

            # 没有工具调用 = 任务完成
            if not llm_response.tool_calls:
                break

            # 执行工具调用
            tool_messages = await tool_node.ainvoke({"messages": messages})
            messages.extend(tool_messages["messages"])

        return last_text
```

### 5.2 `registry.py`：AgentRegistry 单例

**职责**：管理所有已注册的子 Agent，提供 register/resolve 接口。

**用法**：

```python
# 注册
AgentRegistry.register(KnowledgeAgent())

# 查找
agent = AgentRegistry.resolve("knowledge_agent")
result = await agent.run_to_completion("检索XX知识")
```

### 5.3 `knowledge_agent.py`：知识库检索子 Agent

**职责**：从向量库检索相关知识，做 Rerank，返回格式化结果。

**关键配置**：
- `agent_type = "knowledge_agent"`
- `tools = [retrieve_knowledge]`（通过延迟导入赋值，避免循环依赖）
- `system_prompt`：知识检索专家角色提示词
- `model = config.rag_model`（与 retrieve_knowledge 配套）
- `max_turns = 3`（知识检索通常 2 轮足够，3 轮留余量）

**循环导入处理**（重要）：

knowledge_agent.py 不在模块顶层导入 `retrieve_knowledge`，而是在 `__init__` 中延迟导入。原因：

```
app.tools.__init__ (顶部导入 sub_agent_tool)
  → app.agent.sub_agents.__init__ (加载 KnowledgeAgent)
    → knowledge_agent.py (顶层导入 retrieve_knowledge)
      → app.tools.knowledge_tool (需要先加载父包 app.tools)
        → app.tools.__init__ (正在加载中，循环！)
```

延迟到 `__init__` 时，`app.tools` 已加载完成，可安全导入。

### 5.4 `agent_tool.py`：sub_agent_tool LangChain Tool

**职责**：把子 Agent 包装成 LangChain Tool，主 Agent 通过 `subagent_type` 参数选择不同子 Agent。同时包含 P1 修复的可观测性 span 埋点。

**核心逻辑**：

```python
@tool
async def sub_agent_tool(subagent_type: str, prompt: str, description: str = "") -> str:
    agent = AgentRegistry.resolve(subagent_type)
    if agent is None:
        return f"错误：未找到子 Agent 类型 '{subagent_type}'"
    return await agent.run_to_completion(prompt)
```

**P1 修复**：在上述逻辑外包裹 span 埋点（详见第 11 节）。

### 5.5 `__init__.py`：模块入口

**职责**：导出公共 API + 模块加载时自动注册 KnowledgeAgent。

**自动注册机制**：

```python
from .knowledge_agent import KnowledgeAgent

# 模块加载时执行（只执行一次）
AgentRegistry.register(KnowledgeAgent())
```

这样外部模块只需 `from app.agent.sub_agents import sub_agent_tool` 即可使用，不需要手动注册。

---

## 6. 使用方式

### 6.1 方式 1：planner 代码中显式调用

修改后的 [planner.py](file:///d:/oncall2/oncall2/app/agent/aiops/planner.py#L81-L100) 通过 `sub_agent_tool` 调用 KnowledgeAgent：

```python
# 改造前（直接调用工具）：
context_str = await retrieve_knowledge.ainvoke({"query": input_text})

# 改造后（通过子 Agent 调用）：
context_str = await sub_agent_tool.ainvoke({
    "subagent_type": "knowledge_agent",
    "prompt": f"检索与以下任务相关的知识、经验和最佳实践: {input_text}",
    "description": "为规划阶段提供经验支撑"
})
```

**执行流程**：

1. planner 调用 `sub_agent_tool.ainvoke({...})`
2. `sub_agent_tool` 从 AgentRegistry 查找 `knowledge_agent`
3. 调用 `KnowledgeAgent.run_to_completion(prompt)`
4. KnowledgeAgent 在独立上下文中：
   - 第 1 轮：LLM 决定调用 `retrieve_knowledge`
   - 第 2 轮：LLM 看到工具结果，生成最终摘要文本
5. 返回最终文本给 planner
6. planner 把文本作为 `experience_docs` 用于生成计划

### 6.2 方式 2：executor 通过 LLM 自动调用

executor 不需要任何修改！因为它使用的 `DEFAULT_LOCAL_AGENT_TOOLS` 已经自动包含了 `sub_agent_tool`：

```python
# app/tools/__init__.py
DEFAULT_LOCAL_AGENT_TOOLS = (
    retrieve_knowledge,
    get_current_time,
    query_prometheus_alerts,
    sub_agent_tool,  # ← 新增
)
```

executor 把这些工具绑定给 LLM，LLM 在执行步骤时会自行决定是否调用 `sub_agent_tool`。当 LLM 判断"这一步需要查知识库"时，它会生成类似这样的 tool_call：

```json
{
  "name": "sub_agent_tool",
  "args": {
    "subagent_type": "knowledge_agent",
    "prompt": "检索与告警XX相关的知识",
    "description": "为执行步骤提供知识支撑"
  }
}
```

然后 executor 的 ToolNode 自动执行这个 tool_call，调用 KnowledgeAgent 完成检索。

### 6.3 调用日志示例

执行时日志会显示完整的调用链：

```
INFO | planner: 调用 KnowledgeAgent 子 Agent 检索经验...
INFO | agent_tool.sub_agent_tool: AgentTool 调用: type='knowledge_agent', description='为规划阶段提供经验支撑', prompt='检索与XX相关的知识...'
INFO | base.run_to_completion: === SubAgent[knowledge_agent]: run_to_completion 开始 ===
INFO | base.run_to_completion: SubAgent[knowledge_agent] 第 1/3 轮
INFO | base.run_to_completion: SubAgent[knowledge_agent] 调用 1 个工具: ['retrieve_knowledge']
INFO | base.run_to_completion: SubAgent[knowledge_agent] 第 2/3 轮
INFO | base.run_to_completion: SubAgent[knowledge_agent] 第 2 轮无工具调用，任务完成
INFO | base.run_to_completion: SubAgent[knowledge_agent] 执行完成，结果长度: 1234
INFO | agent_tool.sub_agent_tool: AgentTool 完成: type='knowledge_agent', 结果长度=1234
INFO | planner: 找到相关经验文档，长度: 1234
```

---

## 7. 工具过滤防线（防递归嵌套）

### 7.1 为什么要防递归

如果不做限制，子 Agent 可以再调用 `sub_agent_tool` 创建孙 Agent，孙 Agent 再调用 `sub_agent_tool` 创建曾孙 Agent……形成无限嵌套，导致：
- 上下文爆炸（每层都累积消息）
- 资源耗尽（每层都创建 LLM 客户端）
- 死循环（永远不返回）

### 7.2 第 1 层防线：全局禁止列表

在 [base.py](file:///d:/oncall2/oncall2/app/agent/sub_agents/base.py#L34-L39) 定义：

```python
ALL_AGENT_DISALLOWED_TOOLS = frozenset({
    "sub_agent_tool",  # 禁止子 Agent 再 spawn 子 Agent
})
```

### 7.3 过滤执行点

在 [BaseSubAgent.__init__](file:///d:/oncall2/oncall2/app/agent/sub_agents/base.py#L65-L70) 中过滤：

```python
def __init__(self):
    # 第 1 层过滤：剔除全局禁止的工具
    self._filtered_tools = [
        t for t in self.tools
        if t.name not in ALL_AGENT_DISALLOWED_TOOLS
    ]
```

**效果**：即使子 Agent 的 `tools` 列表里包含 `sub_agent_tool`，也会被过滤掉。子 Agent 在 LLM 推理时根本看不到 `sub_agent_tool`，无法调用它，从根源上防止递归。

### 7.4 未来扩展的过滤层

理论文档描述了 4 层过滤，目前实现了第 1 层。未来可以扩展：

```
第 1 层：全局禁止列表 ALL_AGENT_DISALLOWED_TOOLS       ← 已实现
第 2 层：自定义 Agent 额外禁止                          ← 未来扩展
第 3 层：后台 Agent 白名单                              ← 未来扩展（支持后台运行时）
第 4 层：Agent 定义的 tools/disallowedTools             ← 未来扩展（支持 YAML 定义）
```

---

## 8. 如何添加新的子 Agent（扩展指南）

### 8.1 步骤

以添加一个"代码探索 Agent"为例：

**第 1 步**：创建新文件 `app/agent/sub_agents/explore_agent.py`

```python
"""ExploreAgent：代码探索子 Agent"""

from typing import List
from langchain_core.tools import BaseTool

from app.config import config
from .base import BaseSubAgent


class ExploreAgent(BaseSubAgent):
    """代码探索子 Agent

    职责：探索代码库结构，查找功能实现
    工具：[glob_tool, grep_tool, read_tool]（假设这些工具已存在）
    """

    agent_type = "explore_agent"

    when_to_use = (
        "当需要探索代码库结构、查找功能实现、理清调用链时使用。"
        "适用场景：理解项目结构、查找特定函数、分析依赖关系。"
    )

    # 工具列表（如果有循环依赖，参考 knowledge_agent.py 用延迟导入）
    tools: List[BaseTool] = []  # 这里填入实际工具

    system_prompt = """你是一个代码探索专家。你的任务是探索代码库结构。

## 规则
1. 只使用搜索和读取工具，不修改任何文件
2. 尽可能并行发起多个工具调用以提高效率
3. 输出要清晰报告发现，包含关键文件路径
"""
    
    model: str = config.rag_model
    max_turns: int = 10  # 探索任务可能需要多轮搜索
```

**第 2 步**：在 `__init__.py` 中注册

修改 [app/agent/sub_agents/__init__.py](file:///d:/oncall2/oncall2/app/agent/sub_agents/__init__.py)：

```python
from .knowledge_agent import KnowledgeAgent
from .explore_agent import ExploreAgent  # 新增导入

# 注册
AgentRegistry.register(KnowledgeAgent())
AgentRegistry.register(ExploreAgent())  # 新增注册
```

**完成！** 现在 planner 和 executor 都可以通过 `sub_agent_tool` 调用 `explore_agent`：

```python
# planner 显式调用
result = await sub_agent_tool.ainvoke({
    "subagent_type": "explore_agent",
    "prompt": "探索 app/agent 目录的结构",
    "description": "了解项目架构"
})

# executor 通过 LLM 自动调用（LLM 看到 sub_agent_tool 的描述后会自行决定）
```

### 8.2 子 Agent 设计要点

| 设计点 | 说明 |
|--------|------|
| `agent_type` | 全局唯一标识，用于 `subagent_type` 参数 |
| `when_to_use` | 清晰描述适用场景，给 LLM 决策参考 |
| `tools` | 工具白名单，越精简越安全（最小权限原则） |
| `system_prompt` | 角色定义，包含职责、规则、输出格式 |
| `model` | 简单任务用小模型（省钱），复杂任务用大模型 |
| `max_turns` | 知识检索 3 轮够，代码探索可能需要 10 轮 |

### 8.3 循环依赖处理

如果新子 Agent 的工具来自 `app.tools` 且会导致循环导入，参考 [knowledge_agent.py](file:///d:/oncall2/oncall2/app/agent/sub_agents/knowledge_agent.py#L95-L109) 的延迟导入模式：

```python
class YourAgent(BaseSubAgent):
    tools: List[BaseTool] = []  # 类属性留空
    
    def __init__(self):
        # 延迟导入：避免循环依赖
        from app.tools.xxx import some_tool
        self.tools = [some_tool]
        super().__init__()
```

---

## 9. 常见问题排查

### 9.1 子 Agent 调用失败

**现象**：日志显示 `SubAgent[xxx] 执行失败: ...`

**排查**：
1. 检查子 Agent 的 `tools` 是否为空 → 工具导入是否成功
2. 检查 LLM 配置 → `config.dashscope_api_key` 是否有效
3. 检查工具本身是否能正常工作 → 单独测试 `retrieve_knowledge.ainvoke()`

### 9.2 LLM 不调用 sub_agent_tool

**现象**：executor 步骤中 LLM 没有调用 `sub_agent_tool`，而是直接回答或调用了其他工具。

**原因**：LLM 的决策依赖工具的 `description`。如果描述不清晰，LLM 不知道何时该用。

**解决**：
- 检查 `sub_agent_tool` 的 `description` 是否清晰描述了适用场景
- 在 executor 的 SystemMessage 中可以提示"需要知识检索时优先调用 sub_agent_tool"

### 9.3 子 Agent 输出过长

**现象**：子 Agent 返回的文本超过预期，影响主 Agent 的 token 预算。

**解决**：
- 在子 Agent 的 `system_prompt` 中明确输出长度限制（如"输出限制在 800 字以内"）
- 调整 `max_turns` 避免子 Agent 反复检索

### 9.4 循环导入错误

**现象**：`ImportError: cannot import name 'xxx' from partially initialized module`

**原因**：模块加载时的循环依赖。

**解决**：参考 [knowledge_agent.py](file:///d:/oncall2/oncall2/app/agent/sub_agents/knowledge_agent.py#L95-L109) 的延迟导入模式，把工具导入放到 `__init__` 方法中。

### 9.5 Milvus 连接失败

**现象**：导入 `app.tools` 时报 `MilvusException: Fail connecting to server on localhost:19530`

**原因**：Milvus 服务未启动。这是项目原有依赖，不是本次改动引入。

**解决**：启动 Milvus 服务，或检查 `config.milvus_host` 和 `config.milvus_port` 配置。

---

## 10. 与理论文档的映射关系

理论文档《SubAgent 子任务分发》描述了完整的 SubAgent 机制。本次实施按"最小改动"原则，实现了核心子集：

| 理论文档特性 | 本次实施状态 | 说明 |
|-------------|-------------|------|
| Agent 即 Tool | ✅ 已实现 | `sub_agent_tool` 统一工具 |
| 定义式创建模式 | ✅ 已实现 | Python 类形式（非 YAML） |
| Fork 式创建模式 | ❌ 未实现 | 预留扩展点 |
| 上下文隔离 | ✅ 已实现 | 独立 messages 数组 |
| RunToCompletion | ✅ 已实现 | LLM+工具循环 |
| 工具过滤第 1 层（全局禁止） | ✅ 已实现 | `ALL_AGENT_DISALLOWED_TOOLS` |
| 工具过滤第 2 层（自定义禁止） | ❌ 未实现 | 预留扩展点 |
| 工具过滤第 3 层（后台白名单） | ❌ 未实现 | 需要后台运行支持 |
| 工具过滤第 4 层（tools/disallowedTools） | ⚠️ 部分实现 | 通过类属性 `tools` 实现 |
| 后台运行模式 | ❌ 未实现 | 预留扩展点 |
| YAML 定义文件加载 | ❌ 未实现 | 用 Python 类替代 |
| 内置 Agent 类型 | ✅ 部分 | 仅 KnowledgeAgent，未来可加 Explore/Plan |
| TaskManager + task-notification | ❌ 未实现 | 预留扩展点 |

### 10.1 设计取舍说明

**为什么用 Python 类而不是 YAML 定义文件？**

YAML 定义文件的好处是用户可以不写代码就添加子 Agent。但本次实施优先考虑：
- 改动最小：不引入 YAML 解析、文件加载、优先级链等复杂逻辑
- 类型安全：Python 类有 IDE 提示和类型检查
- 调试方便：可以直接断点调试

未来如果子 Agent 数量增多，可以再引入 YAML 加载层。

**为什么不实现 Fork 模式？**

Fork 模式（继承父 Agent 对话历史）的主要场景是 prompt cache 优化。当前项目规模下：
- 主 Agent 上下文不长（planner/executor 各自的 messages 不会太大）
- Qwen 模型的 prompt cache 支持需要单独验证
- Fork 模式实现复杂（需要处理工具调用边界、Fork 标记扫描等）

优先级低于核心的"定义式 + 上下文隔离"。

**为什么不实现后台运行？**

后台运行适合长时间任务（跑测试、安全扫描）。当前 KnowledgeAgent 检索知识通常 2-3 秒完成，前台同步执行即可。未来添加 VerificationAgent（验证 Agent）时再考虑后台模式。

---

## 11. P1 修复：可观测性集成（sub_agent_tool Span 埋点）

### 11.1 问题背景

可观测性设计（[observability-trace-span-metric.md](file:///d:/oncall2/oncall2/docs/superpowers/plans/2026-08-10-observability-trace-span-metric.md)）的 `@trace_node` 装饰器只应用到了 planner/executor/replanner/memory_writer 4 个 LangGraph 节点。本次新增的 `BaseSubAgent.run_to_completion` 没有任何 trace 埋点，导致：

- 子 Agent 的执行耗时被算到 planner/executor 的 span 里，看不到子 Agent 内部细节
- 子 Agent 内部调用的 `retrieve_knowledge` 不会被记录到 `tool_metrics` 表
- 网页端的 `/metrics` 和 `/traces/{id}` 端点看不到子 Agent 的执行记录

### 11.2 修复方案

在 [agent_tool.py](file:///d:/oncall2/oncall2/app/agent/sub_agents/agent_tool.py) 的 `sub_agent_tool` 函数中加 span 埋点：

| 设计点 | 说明 |
|--------|------|
| 埋点位置 | `sub_agent_tool` 函数（统一入口，所有子 Agent 调用都经过这里） |
| `span_type` | `"sub_agent"`（区分普通 `"node"` span） |
| `node_name` | 用 `subagent_type`（如 `"knowledge_agent"`），便于按子 Agent 类型聚合查询 |
| 输入摘要 | JSON 含 `subagent_type` / `description` / `prompt`（各截断） |
| 输出摘要 | 子 Agent 返回的 result 字符串（截断 500 字符） |
| 零开销降级 | 无 trace 上下文时（如单元测试）直接执行，不埋点 |
| 失败记录 | 子 Agent 查找失败或执行异常时，记录 `status="failed"` 的 span |
| 异步保存 | `asyncio.ensure_future` fire-and-forget，不阻塞主流程 |

### 11.3 验证方法

启动服务后触发一次 AIOps 任务，然后查询：

```bash
# 查询最近的 trace，确认含 sub_agent 类型的 span
curl "http://localhost:9900/api/aiops/traces/{trace_id}" | python -m json.tool
```

返回的 `spans` 数组应包含：
- `node_name="planner"` / `span_type="node"`（主流程节点）
- `node_name="knowledge_agent"` / `span_type="sub_agent"`（子 Agent 调用）

### 11.4 与可观测性设计的兼容性

| 设计点 | 兼容性 |
|--------|--------|
| `contextvars` 传播 trace_id | ✅ async 链路自动传播，子 Agent 调用能读到 trace 上下文 |
| `spans` 表结构 | ✅ 复用现有表，`span_type` 字段区分 sub_agent |
| `metrics_collector.record_node_completion` | ✅ 子 Agent 执行计入 node_count，可在 `/metrics` 聚合 |
| `observability_enabled=False` 降级 | ✅ `current_trace()` 返回 None，埋点自动跳过 |

### 11.5 未来扩展（P2）

当前 P1 只记录子 Agent 整体的 span（粗粒度）。未来 P2 可以在 [base.py 的 run_to_completion](file:///d:/oncall2/oncall2/app/agent/sub_agents/base.py#L107-L142) 循环内加细粒度埋点，记录：
- 每轮 LLM 调用的 Token 消耗
- 子 Agent 内部调用的 `retrieve_knowledge` 工具指标（记录到 `tool_metrics` 表，`node_name="knowledge_agent"`）

---

## 附录：改动清单

### 新增文件（6 个）

| 文件 | 职责 |
|------|------|
| `app/agent/sub_agents/__init__.py` | 模块入口 + 自动注册 |
| `app/agent/sub_agents/base.py` | BaseSubAgent 基类 + RunToCompletion |
| `app/agent/sub_agents/registry.py` | AgentRegistry 单例 |
| `app/agent/sub_agents/knowledge_agent.py` | KnowledgeAgent 子类 |
| `app/agent/sub_agents/agent_tool.py` | sub_agent_tool LangChain Tool + P1 span 埋点 |
| `app/agent/sub_agents/README.md` | 本说明文档 |

### 修改文件（2 个）

| 文件 | 改动内容 |
|------|---------|
| `app/tools/__init__.py` | 新增 `sub_agent_tool` 导入，加入 `DEFAULT_LOCAL_AGENT_TOOLS` |
| `app/agent/aiops/planner.py` | 改用 `sub_agent_tool` 调用知识库（替代直接 ainvoke `retrieve_knowledge`） |

### 未修改文件

| 文件 | 原因 |
|------|------|
| `app/agent/aiops/executor.py` | 工具集自动包含 `sub_agent_tool`，LLM 自主决定调用 |
| `app/agent/aiops/replanner.py` | 不涉及子 Agent 调用 |
| `app/agent/aiops/memory_writer.py` | 不涉及子 Agent 调用 |
| `app/agent/aiops/state.py` | 状态结构不变 |

---

*文档版本：1.1 | 最后更新：2026-08-12*
