# L2 转 COSINE 度量切换实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 向量检索从 L2 欧氏距离切换到 COSINE 余弦相似度，适配文本语义检索场景

**Architecture:** 仅改动 3 处搜索逻辑，保留索引创建逻辑（L2 不变）。对于 L2 归一化的 embedding 向量（DashScope text-embedding-v4），L2 和 COSINE 排序等价，索引仍可用。

**Tech Stack:** Milvus IVF_FLAT, pymilvus, langchain-milvus

---

## 背景：L2 vs COSINE 在 Milvus 中的差异

### L2（欧氏距离）
```
公式：d = √(Σ(aᵢ - bᵢ)²)
值域：[0, ∞)
含义：越小越相似，0=完全相同
适用：向量长度也有意义的场景（如图像像素）
```

### COSINE（余弦相似度 → Milvus 中实现为 1−cos(θ)）
```
公式：d = 1 − (a·b)/(|a|·|b|)
值域：[0, 2]
含义：越小越相似，0=方向完全一致，1=正交，2=方向相反
适用：文本语义检索（只关心方向，不关心长度）
```

### 为什么 COSINE 更适合文本检索

文本 embedding 模型（如 text-embedding-v4）输出 L2 归一化向量（||v|| = 1）。此时：
```
L² = 2 − 2cos(θ)   → L 与 cos 单调相关
按 L2 排序 = 按 COSINE 排序（顺序完全一致）
```
COSINE 更符合人的直觉：两个文本"意思相近" → 向量夹角小 → cos 大 → 距离小。

### Milvus 约束

IVF_FLAT 索引创建时指定 `metric_type`，但搜索时的 `search_params.metric_type` 可独立设定。
索引的 metric 影响 IVF 聚类方式，搜索的 metric 影响最终距离计算。
由于 embedding 向量已 L2 归一化，两种 metric 的聚类效果等价，索引无需重建。

### 改动点

| 路径 | 当前 L2 位置 | 改动 |
|---|---|---|
| pymilvus 直接搜索 | `vector_search_service.py:70` | `"L2"` → `"COSINE"` |
| LangChain Milvus 搜索 | `vector_store_manager.py:144` | 加 `param` 参数指定 COSINE |
| 知识工具检索 | `knowledge_tool.py:30-31` | retriever 传参指定 COSINE |

### 不改的文件

| 文件 | 原因 |
|---|---|
| `milvus_client.py` | 索引入库逻辑不动 |
| `vector_embedding_service.py` | embedding 逻辑无关 |
| `config.py` / `.env` | 无需新增配置 |

---

### Task 1: 修改 vector_search_service.py — pymilvus 搜索

**Files:**
- Modify: `app/services/vector_search_service.py:69-72,90`

- [ ] **Step 1: 改 search_params 的 metric_type + 更新注释**

改前：
```python
            # 3. 构建搜索参数
            search_params = {
                "metric_type": "L2",  # 欧氏距离
                "params": {"nprobe": 10},
            }
```
```python
                        score=hit.distance,  # L2 距离，越小越相似
```

改后：
```python
            # 3. 构建搜索参数（COSINE 余弦相似度，适配文本语义检索）
            search_params = {
                "metric_type": "COSINE",  # 余弦距离，越小越相似（0=完全一致）
                "params": {"nprobe": 10},
            }
```
```python
                        score=hit.distance,  # COSINE 距离，越小越相似（0=完全一致）
```

---

### Task 2: 修改 vector_store_manager.py — LangChain Milvus 搜索

**Files:**
- Modify: `app/services/vector_store_manager.py:132-149`

- [ ] **Step 1: 修改 similarity_search 方法，传入 param 指定 COSINE**

改前：
```python
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
```

改后：
```python
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索（使用 COSINE 余弦相似度）

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            docs = self.vector_store.similarity_search(query, k=k, param=search_params)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
```

---

### Task 3: 修改 knowledge_tool.py — 知识工具搜索

**Files:**
- Modify: `app/tools/knowledge_tool.py:29-34`

- [ ] **Step 1: retriever 传参指定 COSINE**

改前：
```python
        # 1. 召回阶段：从向量存储中检索更多文档
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.rag_top_k}
        )
```

改后：
```python
        # 1. 召回阶段：从向量存储中检索更多文档（使用 COSINE 余弦相似度）
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": config.rag_top_k,
                "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
            }
        )
```

---

### Task 4: 创建文档

**Files:**
- Create: `docs/更新日志—向量检索切换COSINE度量.md`

内容：
- L2 与 COSINE 公式/值域/含义对比
- 为什么文本检索用 COSINE
- DashScope embedding 归一化特性说明
- 3 处改动的代码对比
- 各改动的文件路径和行号
