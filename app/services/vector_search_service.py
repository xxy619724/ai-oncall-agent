"""向量检索服务模块"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pymilvus import Collection

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


class SearchResult:
    """搜索结果类"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """向量检索服务 - 负责从 Milvus 中搜索相似向量"""

    def __init__(self):
        """初始化向量检索服务"""
        logger.info("向量检索服务初始化完成")

    @staticmethod
    def _build_expr(
        filter_status: Optional[str],
        filter_confidence: Optional[List[str]],
    ) -> Optional[str]:
        """构造 Milvus 标量过滤表达式

        Args:
            filter_status: 状态过滤，None 表示不过滤；
                          "active" 会转为 `!= 'deprecated'` 兼容无 status 字段的旧数据
            filter_confidence: 置信度白名单，None 表示不过滤

        Returns:
            expr 表达式字符串，None 表示不过滤

        Notes:
            - 用 `!= 'deprecated'` 而非 `== 'active'`：兼容无 status 字段的旧数据（null != 'deprecated' 为真）
            - confidence 用 `in [...]` 语法支持多值白名单
        """
        conditions = []

        # status 过滤：用 != 'deprecated' 兼容旧数据
        if filter_status == "active":
            conditions.append("metadata['status'] != 'deprecated'")
        elif filter_status is not None:
            conditions.append(f"metadata['status'] == '{filter_status}'")

        # confidence 过滤：白名单
        if filter_confidence:
            quoted = ", ".join(f"'{c}'" for c in filter_confidence)
            conditions.append(f"metadata['confidence'] in [{quoted}]")

        if not conditions:
            return None

        return " and ".join(conditions)

    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
        filter_status: Optional[str] = "active",
        filter_confidence: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回最相似的K个结果
            filter_status: 状态过滤，默认 "active" 过滤掉 deprecated；
                           传 None 看全部（查重场景需要）
            filter_confidence: 置信度白名单（如 ["high", "medium"]），None 表示不过滤

        Returns:
            List[SearchResult]: 搜索结果列表

        Raises:
            RuntimeError: 搜索失败时抛出

        Notes:
            - 当 config.milvus_expr_filter_enabled=False 时全局关闭 expr 过滤
            - expr 执行失败自动降级到无过滤重试，保证搜索可用性
        """
        try:
            logger.info(
                f"开始搜索相似文档, 查询: {query}, topK: {top_k}, "
                f"filter_status={filter_status}, filter_confidence={filter_confidence}"
            )

            # 1. 将查询文本向量化
            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功, 维度: {len(query_vector)}")

            # 2. 获取 collection
            collection: Collection = milvus_manager.get_collection()

            # 3. 构建搜索参数（COSINE 余弦相似度，适配文本语义检索）
            search_params = {
                "metric_type": "COSINE",  # 余弦距离，越小越相似（0=完全一致）
                "params": {"nprobe": 10},
            }

            # 4. 构造 expr（受全局开关控制）
            expr = None
            if config.milvus_expr_filter_enabled:
                expr = self._build_expr(filter_status, filter_confidence)
                if expr:
                    logger.debug(f"Milvus expr 过滤: {expr}")

            # 5. 执行搜索（带降级重试）
            try:
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    expr=expr,
                    output_fields=["id", "content", "metadata"],
                )
            except Exception as expr_err:
                if expr is not None:
                    # expr 失败降级到无过滤重试
                    logger.warning(
                        f"Milvus expr 过滤失败，降级无过滤重试: {expr_err}, "
                        f"原 expr={expr}"
                    )
                    results = collection.search(
                        data=[query_vector],
                        anns_field="vector",
                        param=search_params,
                        limit=top_k,
                        expr=None,
                        output_fields=["id", "content", "metadata"],
                    )
                else:
                    raise  # 无 expr 时直接抛出，不降级

            # 6. 解析搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,  # COSINE 距离，越小越相似（0=完全一致）
                        metadata=hit.entity.get("metadata", {}),
                    )
                    search_results.append(result)

            logger.info(f"搜索完成, 找到 {len(search_results)} 个相似文档")
            return search_results

        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            raise RuntimeError(f"搜索失败: {e}") from e


# 全局单例
vector_search_service = VectorSearchService()
