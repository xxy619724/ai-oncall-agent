"""知识检索工具 - 从向量数据库中检索相关信息，并对结果进行重排"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.services.rerank_service import rerank_service


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    流程：召回(向量检索 top_k=10) → 重排(Rerank) → 选 top-3 → 格式化返回

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 1. 召回阶段：从向量存储中检索更多文档（使用 COSINE 余弦相似度）
        # RAG 检索默认过滤 deprecated 经验（用 != 'deprecated' 兼容无 status 字段的旧数据）
        vector_store = vector_store_manager.get_vector_store()
        search_kwargs = {
            "k": config.rag_top_k,
            "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
        }
        if config.milvus_expr_filter_enabled:
            search_kwargs["expr"] = "metadata['status'] != 'deprecated'"
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

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

        logger.info(
            f"重排阶段: 从 {len(docs)} 条中精选 top-{len(reranked_docs)}"
        )

        # 3. 相关性过滤：剔除 rerank 分数低于阈值的低质量文档
        threshold = config.rag_relevance_score_threshold
        filtered_docs = [
            d for d in reranked_docs
            if d.metadata.get("rerank_score", 0) >= threshold
        ]
        if filtered_docs:
            if len(filtered_docs) < len(reranked_docs):
                logger.info(
                    f"相关性过滤: 阈值={threshold}, "
                    f"保留 {len(filtered_docs)}/{len(reranked_docs)} 条"
                )
            reranked_docs = filtered_docs
        else:
            logger.warning(
                f"相关性过滤: 所有文档分数低于阈值 {threshold}，"
                f"保留原始 top-{len(reranked_docs)} 结果"
            )

        # 4. 格式化重排后的文档为上下文
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

        # PDF 分片附带页码范围（提取层注入的 _page_start/_page_end）
        page_start = metadata.get("_page_start")
        page_end = metadata.get("_page_end")
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                source += f" (第{page_start}-{page_end}页)"
            else:
                source += f" (第{page_start}页)"

        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        rerank_score = metadata.get("rerank_score", None)
        score_str = f" (相关性: {rerank_score:.4f})" if rerank_score is not None else ""

        # P1：表格分片带类型标识（提示 LLM 该参考资料是结构化表格）
        type_str = "（表格）" if metadata.get("_type") == "table" else ""

        formatted = f"【参考资料 {i}】{score_str}{type_str}"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
