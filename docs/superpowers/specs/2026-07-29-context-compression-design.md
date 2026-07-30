# 对话上下文压缩功能设计

**目标：** 在长对话和 RAG 多轮检索场景中，当 token 使用量达到模型上下文窗口的 70% 时，自动触发大模型摘要压缩旧对话，保留最近 3 轮原文，避免撑爆上下文窗口。

## 背景问题

### 当前对话管理的 3 个问题

1. **对话历史无限增长**：`MemorySaver` 存储所有历史消息，没有任何 eviction 或压缩机制
2. **系统提示词重复发送**：每次 `query()` / `query_stream()` 都重新传入 `SystemMessage`，导致历史中有 N 份相同 system prompt，浪费 token
3. **无 token 管理**：qwen-max 有 128K 上下文窗口，但从未被监控或管理

### qwen-max 上下文窗口

| 模型 | 上下文窗口 |
|---|---|
| qwen-max (最新) | 131,072 tokens (128K) |
| 70% 阈值 | ~91,750 tokens |
| 压缩后保留 (最近 3 轮) | 约可控制在 20K-30K tokens |

## 技术方案

### 核心组件：LangChain SummarizationMiddleware

LangChain 1.2.10 已内置 `langchain.agents.middleware.SummarizationMiddleware`，完全匹配需求：

| 参数 | 用途 | 本项目配置值 |
|---|---|---|
| `model` | 用于生成摘要的 LLM | ChatQwen (与主 agent 同模型) |
| `trigger` | 触发压缩的条件 | `("fraction", 0.7)` — 上下文 70% 时触发 |
| `keep` | 压缩后保留多少消息 | `("messages", 6)` — 保留最近 3 轮 (6 条) |
| `summary_prompt` | 自定义摘要提示词 | 使用默认 (LangChain 内置) |

**运行原理：**

```
SummarizationMiddleware.before_model()
    │
    ├─ 计算当前消息列表的总 token 数
    ├─ 与 trigger 阈值比较
    │
    ├─ < 70%: 不做任何操作，返回 None（正常流程）
    │
    └─ ≥ 70%:
         ├─ 确定安全截断点（保证 AI/Tool 消息对完整）
         ├─ 分割消息：[旧消息] + [最近 N 条保留消息]
         ├─ 调用 LLM 生成旧消息摘要
         ├─ 构建新消息列表：[摘要] + [最近 N 条保留消息]
         └─ 返回 RemoveMessage + 新消息列表（替换 checkpointer 中的状态）
```

### 修复系统提示词重复

当前代码：
```python
messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]
```

每次调用传入新 SystemMessage → `add_messages` 追加 → 历史中有多份。

**修复方案：** 使用 `create_agent(system_prompt=...)` 参数，让 LangChain 在 agent 初始化时只设置一次 system prompt，后续调用只需传 `HumanMessage`。

### 配置项

新增到 `config.py` / `.env`：

| 配置名 | 默认值 | 说明 |
|---|---|---|
| `rag_context_window_size` | 131072 | qwen-max 上下文窗口大小 |
| `rag_compression_threshold` | 0.7 | 触发压缩的 token 占比阈值 |
| `rag_keep_recent_rounds` | 3 | 保留最近几轮原文 |
| `rag_compression_model` | "qwen-max" | 压缩用的模型 (与主模型同) |

## 架构数据流

```
用户 → /api/chat (query/query_stream)
         │
         ▼
    RagAgentService.query(question, session_id)
         │
         ├─ 1. 初始化 agent（含 SummarizationMiddleware）
         │
         ├─ 2. build messages: [HumanMessage(question)]
         │    （不再传 SystemMessage，由 agent 托管）
         │
         ├─ 3. agent.ainvoke({"messages": messages})
         │    │
         │    ▼
         │    LangGraph 执行：
         │    │
         │    ├─ [NEW] SummarizationMiddleware.before_model()
         │    │    ├─ 检查 token 占比
         │    │    ├─ ≥ 70% → LLM 摘要压缩旧消息
         │    │    └─ 更新状态
         │    │
         │    ├─ Agent 节点（LLM 决策）
         │    ├─ Tool 节点（执行工具）
         │    └─ ... 循环直到生成回答
         │
         └─ 4. 返回回答
```

## 压缩触发流程详解

```
触发条件：token 使用量 / context_window_size ≥ 0.7

Step 1: 计算当前消息总 token
    → model.get_num_tokens_from_messages(state["messages"])

Step 2: 检查是否 ≥ 阈值
    → total_tokens / context_window_size ≥ threshold?

Step 3: 确定安全截断点
    → 在保留最近 N 轮（6 条消息）处截断
    → 保证 AI/Tool 消息对完整（不会在 tool call 中间截断）

Step 4: 分割消息
    → messages_to_summarize = messages[:-6]  # 需要摘要的旧消息
    → preserved_messages = messages[-6:]      # 保留的最近消息

Step 5: 调用 LLM 生成摘要
    → model.invoke(summary_prompt.format(messages=messages_to_summarize))
    → 摘要 prompt：LangChain 内置，包含 "提取最重要的上下文信息"

Step 6: 替换状态
    → 返回 { messages: [RemoveMessage(ALL), summary_msg, *preserved_messages] }
    → checkpointer 中的状态被永久替换
```

## 需修改文件清单

| 文件 | 操作 | 内容 |
|---|---|---|
| `app/config.py` | 修改 | 新增 `rag_context_window_size`、`rag_compression_threshold`、`rag_keep_recent_rounds`、`rag_compression_model` |
| `.env` | 修改 | 新增对应的环境变量 |
| `app/services/rag_agent_service.py` | 修改 | 注入 `SummarizationMiddleware` + 修复 system prompt 重复 |
| `docs/更新日志—上下文压缩功能.md` | 新建 | 详细教程文档 |

## 不变的部分

- `app/tools/knowledge_tool.py` — 不需要修改
- `app/services/rerank_service.py` — 不需要修改
- `app/services/vector_store_manager.py` — 不需要修改
- `app/api/chat.py` — 接口无变化
- `pyproject.toml` — 依赖不变 (langchain 1.2.10 已满足)
