# 更新日志 — 向量检索切换 COSINE 度量

## 功能概述

将向量检索的距离度量从 **L2（欧氏距离）** 切换为 **COSINE（余弦相似度）**，更适配文本语义检索场景。

---

## 先搞懂 L2 和 COSINE 的区别

### L2 欧氏距离

L2 计算的是两个向量在**空间中的直线距离**：

```
公式：d(a, b) = √[(a₁−b₁)² + (a₂−b₂)² + ... + (aₙ−bₙ)²]

值域：[0, ∞)
含义：越小越相似，0 = 两个向量完全重合
```

**缺点**：向量长度（模长）也会影响距离。比如两段文本意思相同但长度不同，L2 距离可能很大。

### COSINE 余弦相似度

COSINE 计算的是两个向量**方向的夹角余弦**：

```
公式：cos(θ) = (a·b) / (|a|·|b|)

在 Milvus 中存储为 1−cos(θ)（距离形式）：
d = 1 − cos(θ)

值域：[0, 2]
含义：越小越相似
  0 = 方向完全一致（最相似）
  1 = 正交（不相关）
  2 = 方向相反
```

**优点**：只关注方向的相似性，忽略向量长度。两段同义文本即使详略不同，COSINE 距离仍然很小。

### 为什么 COSINE 更适合文本检索

文本 embedding 模型（比如我们用的 `text-embedding-v4`）输出的向量是 **L2 归一化**的：

```
向量长度 = 1（每个向量的模长都标准化为 1）
```

在归一化向量上，L2 和 COSINE 的关系是：

```
L² = 2 − 2cos(θ)

所以：按 L2 排序 = 按 COSINE 排序（顺序完全一致）
```

但 COSINE 更符合语义直觉——我们说"两段文本意思相近"，本质上是它们在语义空间中的**方向一致**，与文本长度无关。

---

## 为什么可以不改索引

Milvus 的 IVF_FLAT 索引在创建时指定了 `metric_type: "L2"`（`milvus_client.py:198`）。

但搜索时的 `search_params.metric_type` 可以独立设定：

| 阶段 | metric_type | 作用 |
|---|---|---|
| 索引创建 | L2 | IVF 聚类时划分向量空间 |
| 搜索 | COSINE | 最终距离计算 |

由于向量已 L2 归一化，两种 metric 的**聚类效果等价**（同一簇内的向量也是 COSINE 相近的向量），所以索引无需重建。这就是"仅改检索逻辑，不改索引入库逻辑"可行的原因。

---

## 改了哪些文件

### 改动 1：`app/services/vector_search_service.py`

pymilvus 直接搜索（用于 AIOps 诊断等场景）。

**改前：**
```python
search_params = {
    "metric_type": "L2",       # 欧氏距离
    "params": {"nprobe": 10},
}
```
```python
score=hit.distance,  # L2 距离，越小越相似
```

**改后：**
```python
search_params = {
    "metric_type": "COSINE",   # 余弦距离，越小越相似（0=完全一致）
    "params": {"nprobe": 10},
}
```
```python
score=hit.distance,  # COSINE 距离，越小越相似（0=完全一致）
```

### 改动 2：`app/services/vector_store_manager.py`

LangChain Milvus 的 `similarity_search` 方法，用于 `vector_store_manager` 的搜索。

**改前：**
```python
def similarity_search(self, query, k=3):
    docs = self.vector_store.similarity_search(query, k=k)
```

**改后：**
```python
def similarity_search(self, query, k=3):
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    docs = self.vector_store.similarity_search(query, k=k, param=search_params)
```

加了一个 `param=search_params` 参数，显式告诉 Milvus 用 COSINE 算距离。

### 改动 3：`app/tools/knowledge_tool.py`

知识工具中的 retriever（RAG 检索的核心路径）。

**改前：**
```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": config.rag_top_k}
)
```

**改后：**
```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": config.rag_top_k,
        "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
    }
)
```

---

## 没有改动的文件

| 文件 | 原因 |
|---|---|
| `app/core/milvus_client.py` | 索引创建逻辑不动（L2 仍可正常聚类） |
| `app/config.py` + `.env` | 无需新增配置 |
| `app/services/vector_embedding_service.py` | embedding 逻辑无关 |

---

## 验证方法

启动后上传文档做索引，然后问问题测试。观察日志和结果：

1. pymilvus 搜索路径（AIOps 等）：无日志变化，但检索结果应当更符合语义
2. LangChain 搜索路径（RAG 对话）：同上
3. 查看 `SearchResult.score` 值域：
   - 原来 L2：一般在 0~50 之间
   - 现在 COSINE：一般在 0~1 之间（0=最相似）

如果启动时报错 `Metric type mismatch`，说明 Milvus 版本强制要求搜索 metric 匹配索引 metric。此时回退方案：保留 L2 search_params，只改 score 解释。但这种情况很少见。
