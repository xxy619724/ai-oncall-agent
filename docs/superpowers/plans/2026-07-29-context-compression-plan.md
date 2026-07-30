# 对话上下文压缩功能实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在对话 token 达到 qwen-max 上下文窗口 70% 时，自动触发 LLM 摘要压缩旧消息，保留最近 3 轮原文

**Architecture:** 利用 LangChain 1.2.10 内置的 `SummarizationMiddleware`（`langchain.agents.middleware`），注入到 `create_agent(middleware=[...])` 中。同时修复系统提示词重复发送的 bug。

**Tech Stack:** Python, LangChain 1.2.10, LangGraph, ChatQwen

---

### Task 1: 配置变更 — config.py + .env

**Files:**
- Modify: `app/config.py`
- Modify: `.env`

- [ ] **Step 1: config.py 增加上下文压缩配置字段**

在 `Settings` 类的 `# 重排（Rerank）配置` 后面增加：

```python
    # 上下文压缩配置
    rag_context_window_size: int = 131072  # qwen-max 上下文窗口（128K tokens）
    rag_compression_threshold: float = 0.7  # 触发压缩的 token 占比阈值
    rag_keep_recent_rounds: int = 3  # 保留最近几轮完整原文
    rag_compression_model: str = "qwen-max"  # 压缩用的模型
```

- [ ] **Step 2: .env 新增环境变量**

```ini
# 上下文压缩配置
RAG_CONTEXT_WINDOW_SIZE=131072
RAG_COMPRESSION_THRESHOLD=0.7
RAG_KEEP_RECENT_ROUNDS=3
RAG_COMPRESSION_MODEL=qwen-max
```

---

### Task 2: 修改 rag_agent_service.py — 注入 SummarizationMiddleware + 修复系统提示词

**Files:**
- Modify: `app/services/rag_agent_service.py`

需要改 3 处：

1. 导入 `SummarizationMiddleware`
2. 在 `_initialize_agent()` 中注入 middleware
3. 在 `query()` 和 `query_stream()` 中移除重复的 `SystemMessage`（修复 bug）

- [ ] **Step 1: 在文件顶部增加导入**

```python
# 在现有导入下面增加：
from langchain.agents.middleware import SummarizationMiddleware
```

- [ ] **Step 2: 修改 `_initialize_agent()`，注入 SummarizationMiddleware**

```python
    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具和 SummarizationMiddleware）"""
        if self._agent_initialized:
            return

        for name, server in config.mcp_servers.items():
            hint = suggest_mcp_transport(
                str(server.get("url", "")),
                str(server.get("transport", "")),
            )
            if hint:
                logger.warning(f"MCP 配置 [{name}]: {hint}")

        mcp_client = await get_mcp_client_with_retry()
        mcp_tools, mcp_err = await load_mcp_tools_safe(mcp_client)
        if mcp_err:
            logger.warning(
                f"MCP 工具加载失败，将仅使用本地工具继续运行:\n{mcp_err}"
            )
            self.mcp_tools = []
        else:
            self.mcp_tools = mcp_tools
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        all_tools = self.tools + self.mcp_tools

        # 构建 SummarizationMiddleware（上下文自动压缩）
        summarization_middleware = SummarizationMiddleware(
            model=self.model,
            trigger=("fraction", config.rag_compression_threshold),
            keep=("messages", config.rag_keep_recent_rounds * 2),
        )

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
            system_prompt=self.system_prompt,  # 让 LangChain 托管 system prompt
            middleware=[summarization_middleware],
        )

        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")
```

- [ ] **Step 3: 修改 `query()` 方法，移除重复 SystemMessage**

```python
    async def query(self, question: str, session_id: str) -> str:
        """..."""
        try:
            await self._initialize_agent()
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 只需要传 HumanMessage，system prompt 由 agent 托管
            messages = [HumanMessage(content=question)]

            agent_input = {"messages": messages}
            config_dict = {"configurable": {"thread_id": session_id}}

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案...（后续代码不变）
```

- [ ] **Step 4: 修改 `query_stream()` 方法，同样移除重复 SystemMessage**

```python
    async def query_stream(self, question: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """..."""
        try:
            await self._initialize_agent()
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 只需要传 HumanMessage，system prompt 由 agent 托管
            messages = [HumanMessage(content=question)]

            agent_input = {"messages": messages}
            config_dict = {"configurable": {"thread_id": session_id}}

            # 后续流式处理代码不变...
```

---

### Task 3: 编写文档 — 更新日志

**Files:**
- Create: `docs/更新日志—上下文压缩功能.md`

- [ ] **Step 1: 创建更新日志文档**

```markdown
# 更新日志 — 上下文压缩功能

## 功能概述

在长对话和 RAG 多轮检索场景中，当 token 使用量达到模型上下文窗口的 70% 时，自动触发大模型摘要压缩旧对话，保留最近 3 轮原文，避免上下文窗口被撑爆。

## 为什么需要这个功能

### 原来的问题

1. **对话历史无限增长**：每轮对话的消息都存储在 MemorySaver 中，从不清理。经过 N 轮对话后，消息数量持续膨胀。
2. **系统提示词重复**：每次调用 `query()` 都重新传入 `SystemMessage`，但 LangGraph 的 `add_messages` 会追加而不是替换。经过 5 轮对话后，消息列表变成：
   ```
   [sys, user1, ai1, sys, user2, ai2, sys, user3, ai3, sys, user4, ai4, sys, user5, ai5]
   ```
   5 份 system prompt 浪费了大量 token！
3. **RAG 多轮检索叠加**：每轮 RAG 检索会带回 3 个参考文档（每个 ~800 tokens）+ rerank 信息，多轮对话更容易撑爆上下文。

### qwen-max 的上下文窗口

| 模型 | 上下文窗口 |
|---|---|
| qwen-max | 131,072 tokens (128K) |
| 70% 阈值 | ~91,750 tokens |
| 压缩后保留 | 最近 3 轮（约 15K-30K tokens） |

一轮典型对话的 token 消耗：
- User 问题：~100 tokens
- AI 回答：~200-500 tokens
- RAG 参考文档：~2,400 tokens (3个 × 800)
- 合计一轮：~3,000 tokens

所以约 30 轮对话就会达到 70% 阈值。

## 实现原理

### 使用 LangChain 内置的 SummarizationMiddleware

LangChain 1.2.10 已经内置了 `langchain.agents.middleware.SummarizationMiddleware`，我们直接使用它。

### 它是如何工作的？

SummarizationMiddleware 是一个 `AgentMiddleware`，在 agent 每次调用 LLM **之前**（`before_model` 钩子）检查消息的 token 总量：

```
agent.ainvoke({"messages": [...]})
    │
    ▼
[SummarizationMiddleware.before_model()]
    │
    ├─ 计算当前消息总 token 数
    │   → model.get_num_tokens_from_messages(state["messages"])
    │
    ├─ 计算占比：total_tokens / context_window_size ≥ threshold?
    │
    ├─ < 70%: 什么都不做，返回 None（正常流程）
    │
    └─ ≥ 70%: 触发压缩
         ├─ 确定安全截断点（保证不截断 AI/Tool 消息对）
         ├─ 分割：messages_to_summarize = 旧消息
         │       preserved_messages = 最近 3 轮（6 条消息）
         ├─ 调用 LLM 生成摘要：
         │   "Summarize the following conversation history..."
         ├─ 构建新消息列表：[summary_message] + preserved_messages
         └─ 返回 RemoveMessage + 新列表 → 替换 checkpointer 状态
```

### 配置参数

| 参数 | 本项目配置值 | 说明 |
|---|---|---|
| `model` | ChatQwen (qwen-max) | 用于生成摘要的模型 |
| `trigger` | `("fraction", 0.7)` | 上下文窗口占比 ≥ 70% 时触发 |
| `keep` | `("messages", 6)` | 压缩后保留最近 6 条消息（3 轮） |

`("fraction", ...)` 模式会自动读取 `model.profile.max_input_tokens`（qwen-max = 131072）。

### 同时修复的系统提示词重复 bug

**改前：**
```python
# 每次 query() 都传 SystemMessage
messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]
```

**改后：**
```python
# system prompt 在 create_agent 时设置一次，后续只需传用户消息
create_agent(..., system_prompt=self.system_prompt)
# ...
messages = [HumanMessage(content=question)]
```

## 修改的文件

### 1. `app/config.py` — 新增配置

```python
rag_context_window_size: int = 131072    # qwen-max 上下文窗口
rag_compression_threshold: float = 0.7   # 70% 触发压缩
rag_keep_recent_rounds: int = 3          # 保留最近 3 轮
rag_compression_model: str = "qwen-max"  # 压缩模型
```

### 2. `.env` — 新增环境变量

```ini
RAG_CONTEXT_WINDOW_SIZE=131072
RAG_COMPRESSION_THRESHOLD=0.7
RAG_KEEP_RECENT_ROUNDS=3
RAG_COMPRESSION_MODEL=qwen-max
```

### 3. `app/services/rag_agent_service.py` — 核心修改

**改前：**
```python
from langchain.agents import create_agent

class RagAgentService:
    async def _initialize_agent(self):
        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

    async def query(self, question, session_id):
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]
        result = await self.agent.ainvoke({"messages": messages}, ...)
```

**改后：**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware  # 新增

class RagAgentService:
    async def _initialize_agent(self):
        summarization_middleware = SummarizationMiddleware(
            model=self.model,
            trigger=("fraction", config.rag_compression_threshold),
            keep=("messages", config.rag_keep_recent_rounds * 2),
        )
        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
            system_prompt=self.system_prompt,  # 新增：LangChain 托管 system prompt
            middleware=[summarization_middleware],  # 新增：注入压缩中间件
        )

    async def query(self, question, session_id):
        # 不再传 SystemMessage，由 agent 托管
        messages = [HumanMessage(content=question)]
        result = await self.agent.ainvoke({"messages": messages}, ...)
```

## 技术细节

### 为什么不自己实现而是用 LangChain 的？

1. **省代码**：`SummarizationMiddleware` 已经处理了安全截断点（确保 AI/Tool 消息对完整）、token 计数、摘要 prompt 等复杂逻辑
2. **省维护**：LangChain 社区维护，有 bug 修复（比如 #34607 修复了 token 膨胀问题）
3. **开箱即用**：直接 `from langchain.agents.middleware import SummarizationMiddleware` 即可

### SummarizationMiddleware 的 `trigger` 模式

支持多种触发模式：

| 模式 | 写法 | 说明 |
|---|---|---|
| 基于 token 占比 | `("fraction", 0.7)` | 上下文窗口占比 ≥ 70% 时触发（推荐，自动适配任何模型） |
| 基于 token 数量 | `("tokens", 90000)` | 总 token ≥ 90,000 时触发 |
| 基于消息数量 | `("messages", 50)` | 消息数量 ≥ 50 条时触发 |
| 组合条件 (AND) | `{"tokens": 5000, "messages": 10}` | 两个条件都满足时才触发 |
| 组合条件 (OR) | `[("fraction", 0.8), ("messages", 100)]` | 任一满足即触发 |

本项目使用 `("fraction", 0.7)`，自动适配模型上下文窗口。

## 验证方法

启动应用后，与 AI 进行多轮对话（特别是 RAG 检索对话），观察日志：

### 正常情况（token < 70%）
没有压缩日志，对话正常进行。

### 触发压缩（token ≥ 70%）
SummarizationMiddleware 内部会在触发时记录日志（LangChain 内部 debug 日志）。你可以通过以下方式验证：

1. **多轮对话测试**：连续问 30+ 轮问题，观察响应是否正常
2. **长上下文测试**：先上传一个大文档做 RAG 索引，然后连续问多轮相关问题
3. **查看日志**：如果 `LOG_LEVEL=DEBUG`，可以看到 Middleware 的触发情况
