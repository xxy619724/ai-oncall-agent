# 重排（Rerank）功能实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在召回文档后，使用阿里云百炼的 rerank 模型对文档进行重排，提升最终生成回答的质量

**Architecture:** 当前流程 `向量检索(top_k=3) → 生成` 改为 `向量检索(top_k=10) → Rerank重排 → 选top-3 → 生成`。新增 `rerank_service.py` 调用 DashScope rerank API，修改 `knowledge_tool.py` 集成重排步骤，修改 `config.py/.env` 增加重排相关配置。

**Tech Stack:** Python, DashScope (gte-rerank), LangChain, LangGraph

---

### Task 1: 配置变更 — config.py + .env

**Files:**
- Modify: `app/config.py`
- Modify: `.env`

- [ ] **Step 1: config.py 增加重排配置字段**

```python
# 在 Settings 类中 RAG 配置区域增加以下字段：

    # RAG 配置
    rag_top_k: int = 10  # 改为初始检索数量（默认10条）
    rag_model: str = "qwen-max"

    # 重排（Rerank）配置
    rag_rerank_top_k: int = 3  # 重排后保留的数量
    rag_rerank_model: str = "gte-rerank"  # 百炼 rerank 模型

    # ... 原有配置保持不变
```

- [ ] **Step 2: .env 增加重排环境变量**

```ini
# RAG 配置
RAG_TOP_K=10          # 从 3 改为 10（召回阶段查更多文档）
RAG_MODEL=qwen-max

# 重排（Rerank）配置
RAG_RERANK_TOP_K=3
RAG_RERANK_MODEL=gte-rerank
```

---

### Task 2: 创建重排服务 — rerank_service.py

**Files:**
- Create: `app/services/rerank_service.py`

- [ ] **Step 1: 创建 rerank_service.py**

```python
"""重排服务模块 - 使用阿里云百炼 rerank 模型对召回文档进行语义重排"""

from typing import List
from openai import OpenAI
from loguru import logger
from langchain_core.documents import Document

from app.config import config


class RerankService:
    """重排服务 - 调用 DashScope rerank API 对文档进行语义重排序"""

    def __init__(self):
        self.api_key = config.dashscope_api_key
        self.model = config.rag_rerank_model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        logger.info(f"Rerank 服务初始化完成, model={self.model}")

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        对召回文档进行重排，返回重排后得分最高的 top_k 个文档

        Args:
            query: 用户查询
            documents: 召回的文档列表
            top_k: 返回的文档数量

        Returns:
            List[Document]: 重排后的文档列表（已按相关性降序排列）
        """
        if not documents:
            return []

        try:
            logger.info(f"Rerank 开始: query='{query}', 输入文档数={len(documents)}, top_k={top_k}")

            docs_to_rerank = documents[:top_k * 3] if top_k > 0 else documents

            texts = [doc.page_content for doc in docs_to_rerank]

            response = self.client.rerank.create(
                model=self.model,
                query=query,
                documents=texts,
            )

            results = response.results
            sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)

            reranked_docs = []
            for r in sorted_results[:top_k]:
                idx = r.index
                doc = docs_to_rerank[idx]
                doc.metadata["rerank_score"] = r.relevance_score
                reranked_docs.append(doc)

            for i, doc in enumerate(reranked_docs):
                logger.debug(f"  Rerank #{i+1}: score={doc.metadata.get('rerank_score', 'N/A'):.4f}, content_preview={doc.page_content[:60]}...")

            logger.info(f"Rerank 完成: 输出文档数={len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            logger.warning("Rerank 失败，降级为使用原始排序结果")
            return documents[:top_k]


# 全局单例
rerank_service = RerankService()
```

- [ ] **Step 2: 在 `app/services/__init__.py` 中导出 rerank_service**

检查 `app/services/__init__.py`，确保 rerank_service 可被导入。

---

### Task 3: 修改 knowledge_tool — 集成重排

**Files:**
- Modify: `app/tools/knowledge_tool.py`

- [ ] **Step 1: 修改 knowledge_tool.py，在召回后调用重排**

```python
"""知识检索工具 - 从向量数据库中检索相关信息，并对结果进行重排"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.services.rerank_service import rerank_service  # 新增导入


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 1. 召回阶段：从向量存储中检索更多文档（top_k=10）
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.rag_top_k}
        )

        docs = retriever.invoke(query)

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        logger.info(f"召回阶段: 检索到 {len(docs)} 个相关文档")

        # 2. 重排阶段：使用 rerank 模型对文档进行语义重排
        reranked_docs = rerank_service.rerank(
            query=query,
            documents=docs,
            top_k=config.rag_rerank_top_k,
        )

        logger.info(f"重排阶段: 从 {len(docs)} 条中精选 top-{len(reranked_docs)}")

        # 3. 格式化重排后的文档为上下文
        context = format_docs(reranked_docs)

        return context, reranked_docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        rerank_score = metadata.get("rerank_score", None)
        score_str = f" (相关性: {rerank_score:.4f})" if rerank_score is not None else ""

        formatted = f"【参考资料 {i}】{score_str}"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
```

---

### Task 4: 编写文档 — 更新日志

**Files:**
- Create: `docs/更新日志—重排功能.md`

- [ ] **Step 1: 创建更新日志文档**

```markdown
# 更新日志 — 重排功能

## 功能概述

在 RAG（检索增强生成）流程中增加了**重排（Rerank）** 步骤，提升最终生成回答的质量和相关性。

## 原有流程

```
用户问题 → 向量检索(top_k=3) → 直接生成回答
```

问题：向量检索只返回3条，精度不够，可能漏掉更相关的文档。

## 新流程（三段式）

```
用户问题 → 1.召回(向量检索 top_k=10) → 2.重排(Rerank) → 3.生成(选top_k=3 + LLM)
```

### 1. 召回（Retrieval）
- 将用户问题向量化，去 Milvus 向量数据库进行 ANN 相似度搜索
- 查询最相似的 **10 个**片段（`RAG_TOP_K=10`）
- 特点：快但准度低

### 2. 重排（Rerank）
- 使用阿里云百炼平台的重排模型 `gte-rerank`
- 逐对计算用户问题与每个召回片段的语义相关性
- 按相关性分数降序排列
- 特点：慢但准度高

### 3. 生成（Generation）
- 从重排结果中选出最相关的 **3 个**片段（`RAG_RERANK_TOP_K=3`）
- 将 3 个片段 + 用户问题一起交给大模型
- 大模型根据提供的片段进行回答

## 修改的文件

### 1. `app/config.py`
- 新增重排配置字段：
  - `rag_rerank_top_k: int = 3` — 重排后保留的文档数
  - `rag_rerank_model: str = "gte-rerank"` — 百炼 rerank 模型名

### 2. `.env`
- `RAG_TOP_K` 从 `3` 改为 `10` — 召回阶段查更多文档供重排
- 新增 `RAG_RERANK_TOP_K=3` — 重排后保留 top-3
- 新增 `RAG_RERANK_MODEL=gte-rerank` — 指定 rerank 模型

### 3. `app/services/rerank_service.py`（新建）
- 封装 `RerankService` 类
- 通过 DashScope OpenAI 兼容接口调用 `rerank.create()` API
- 输入：用户查询 + 文档列表
- 输出：按语义相关性降序排列的文档列表
- 包含异常处理，失败时降级使用原始排序

### 4. `app/tools/knowledge_tool.py`
- 修改 `retrieve_knowledge` 函数
- 原本：`向量检索 → 格式化 → 返回`
- 改为：`向量检索(top_k=10) → Rerank重排 → 选top-3 → 格式化 → 返回`
- format_docs 中新增显示 rerank 相关性分数
```

---

## 执行方式

请选择以下一种执行方式：

**1. 逐步指导（推荐）** — 我逐一执行每个 Task，每完成一步跟你确认

**2. 全部自动执行** — 我一次性完成所有修改
