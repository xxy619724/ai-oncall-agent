"""重排服务模块 - 使用阿里云百炼 rerank 模型对召回文档进行语义重排"""

from typing import List

import httpx
from langchain_core.documents import Document
from loguru import logger

from app.config import config


RERANK_API_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


class RerankService:
    """重排服务 - 调用 DashScope rerank API 对文档进行语义重排序"""

    def __init__(self):
        self.api_key = config.dashscope_api_key
        self.model = config.rag_rerank_model
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
            logger.info(
                f"Rerank 开始: query='{query[:60]}...', "
                f"输入文档数={len(documents)}, top_k={top_k}"
            )

            texts = [doc.page_content for doc in documents]

            payload = {
                "model": self.model,
                "query": query,
                "documents": texts,
                "top_n": top_k,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            with httpx.Client(timeout=60) as client:
                response = client.post(RERANK_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            results = result.get("output", {}).get("results", [])
            sorted_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

            reranked_docs = []
            for r in sorted_results:
                idx = r.get("index", 0)
                if idx < len(documents):
                    doc = documents[idx]
                    doc.metadata["rerank_score"] = r.get("relevance_score", 0.0)
                    reranked_docs.append(doc)

            for i, doc in enumerate(reranked_docs):
                logger.debug(
                    f"  Rerank #{i+1}: score={doc.metadata.get('rerank_score', 'N/A')}, "
                    f"preview={doc.page_content[:60]}..."
                )

            logger.info(f"Rerank 完成: 输出文档数={len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            logger.warning("Rerank 失败，降级为使用原始排序结果")
            return documents[:top_k]


# 全局单例
rerank_service = RerankService()
