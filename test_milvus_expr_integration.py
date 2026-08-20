"""Milvus expr 标量过滤集成测试

用 mock 验证 search_similar_documents 端到端传递 expr 到 collection.search，
不依赖真实 Milvus 服务和 Embedding API。

覆盖场景：
1. 默认 active 过滤 → collection.search 收到 expr
2. filter_status=None → collection.search 收到 expr=None
3. confidence 过滤 → expr 包含 confidence 条件
4. expr 失败降级重试 → 第二次 search 不带 expr
5. 全局开关关闭 → expr 不传递
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Windows 控制台 GBK 编码兼容
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from app.config import config
from app.services.vector_search_service import VectorSearchService


def _make_mock_hit(entity_id, content, metadata):
    """构造 mock search hit"""
    hit = MagicMock()
    data = {"id": entity_id, "content": content, "metadata": metadata}
    hit.entity.get = MagicMock(side_effect=lambda k, default=None: data.get(k, default))
    hit.distance = 0.1
    return hit


def _make_mock_collection(hits):
    """构造 mock collection，返回指定 hits"""
    collection = MagicMock()
    collection.search.return_value = [hits]
    return collection


def test_default_filter_passes_expr():
    """场景1：默认 filter_status='active' → collection.search 收到 expr"""
    service = VectorSearchService()

    hits = [_make_mock_hit("1", "内容", {"status": "active"})]
    mock_collection = _make_mock_collection(hits)

    with patch("app.services.vector_search_service.milvus_manager") as mock_mgr, \
         patch("app.services.vector_search_service.vector_embedding_service") as mock_embed:
        mock_mgr.get_collection.return_value = mock_collection
        mock_embed.embed_query.return_value = [0.1] * 1024

        service.search_similar_documents("测试查询", top_k=3)

        # 验证 collection.search 被调用时带了 expr
        call_kwargs = mock_collection.search.call_args.kwargs
        assert call_kwargs.get("expr") == "metadata['status'] != 'deprecated'", \
            f"默认过滤 expr 未传递: {call_kwargs.get('expr')}"
        print(f"✓ 场景1 通过: 默认 active 过滤 → expr={call_kwargs['expr']}")


def test_none_filter_no_expr():
    """场景2：filter_status=None → collection.search 收到 expr=None"""
    service = VectorSearchService()

    hits = [_make_mock_hit("1", "内容", {"status": "deprecated"})]
    mock_collection = _make_mock_collection(hits)

    with patch("app.services.vector_search_service.milvus_manager") as mock_mgr, \
         patch("app.services.vector_search_service.vector_embedding_service") as mock_embed:
        mock_mgr.get_collection.return_value = mock_collection
        mock_embed.embed_query.return_value = [0.1] * 1024

        service.search_similar_documents("测试查询", top_k=3, filter_status=None)

        call_kwargs = mock_collection.search.call_args.kwargs
        assert call_kwargs.get("expr") is None, \
            f"None 过滤不应传 expr: {call_kwargs.get('expr')}"
        print(f"✓ 场景2 通过: None 过滤 → expr=None")


def test_confidence_filter_passes_expr():
    """场景3：confidence 过滤 → expr 包含 confidence 条件"""
    service = VectorSearchService()

    hits = [_make_mock_hit("1", "内容", {"confidence": "high"})]
    mock_collection = _make_mock_collection(hits)

    with patch("app.services.vector_search_service.milvus_manager") as mock_mgr, \
         patch("app.services.vector_search_service.vector_embedding_service") as mock_embed:
        mock_mgr.get_collection.return_value = mock_collection
        mock_embed.embed_query.return_value = [0.1] * 1024

        service.search_similar_documents(
            "测试查询", top_k=3, filter_confidence=["high", "medium"]
        )

        call_kwargs = mock_collection.search.call_args.kwargs
        expr = call_kwargs.get("expr")
        assert expr is not None and "confidence" in expr, \
            f"confidence 过滤 expr 未传递: {expr}"
        print(f"✓ 场景3 通过: confidence 过滤 → expr={expr}")


def test_expr_failure_fallback():
    """场景4：expr 失败 → 降级重试不带 expr"""
    service = VectorSearchService()

    hits = [_make_mock_hit("1", "内容", {"status": "active"})]
    mock_collection = MagicMock()
    # 第一次带 expr 失败，第二次不带 expr 成功
    mock_collection.search.side_effect = [
        Exception("Milvus expr 语法错误"),
        [hits],  # 降级重试成功
    ]

    with patch("app.services.vector_search_service.milvus_manager") as mock_mgr, \
         patch("app.services.vector_search_service.vector_embedding_service") as mock_embed:
        mock_mgr.get_collection.return_value = mock_collection
        mock_embed.embed_query.return_value = [0.1] * 1024

        results = service.search_similar_documents("测试查询", top_k=3)

        # 验证 search 被调用 2 次（第一次带 expr 失败，第二次降级）
        assert mock_collection.search.call_count == 2, \
            f"应调用 2 次 search，实际 {mock_collection.search.call_count} 次"

        # 第二次调用的 expr 应该是 None（降级）
        second_call_kwargs = mock_collection.search.call_args_list[1].kwargs
        assert second_call_kwargs.get("expr") is None, \
            f"降级重试应 expr=None: {second_call_kwargs.get('expr')}"

        # 验证返回了结果
        assert len(results) == 1, f"降级后应返回 1 条结果，实际 {len(results)}"
        print(f"✓ 场景4 通过: expr 失败降级重试成功，返回 {len(results)} 条结果")


def test_global_switch_disabled():
    """场景5：全局开关关闭 → expr 不传递"""
    service = VectorSearchService()

    hits = [_make_mock_hit("1", "内容", {"status": "active"})]
    mock_collection = _make_mock_collection(hits)

    original = config.milvus_expr_filter_enabled
    try:
        config.milvus_expr_filter_enabled = False

        with patch("app.services.vector_search_service.milvus_manager") as mock_mgr, \
             patch("app.services.vector_search_service.vector_embedding_service") as mock_embed:
            mock_mgr.get_collection.return_value = mock_collection
            mock_embed.embed_query.return_value = [0.1] * 1024

            # 即使传了 filter_status="active"，开关关闭也不应传 expr
            service.search_similar_documents("测试查询", top_k=3, filter_status="active")

            call_kwargs = mock_collection.search.call_args.kwargs
            assert call_kwargs.get("expr") is None, \
                f"全局开关关闭应 expr=None: {call_kwargs.get('expr')}"
            print(f"✓ 场景5 通过: 全局开关关闭 → expr=None（即使传了 filter_status='active'）")
    finally:
        config.milvus_expr_filter_enabled = original


def test_query_results_parsed_correctly():
    """场景6：验证搜索结果正确解析（含 metadata 字段）"""
    service = VectorSearchService()

    metadata = {
        "_source": "aiops_experience",
        "confidence": "high",
        "status": "active",
        "ttl_days": 90,
        "experience_id": "exp_001",
    }
    hits = [_make_mock_hit("exp_001", "CPU 处理方案", metadata)]
    mock_collection = _make_mock_collection(hits)

    with patch("app.services.vector_search_service.milvus_manager") as mock_mgr, \
         patch("app.services.vector_search_service.vector_embedding_service") as mock_embed:
        mock_mgr.get_collection.return_value = mock_collection
        mock_embed.embed_query.return_value = [0.1] * 1024

        results = service.search_similar_documents("CPU 告警", top_k=1)

        assert len(results) == 1
        result = results[0]
        assert result.id == "exp_001"
        assert result.content == "CPU 处理方案"
        assert result.metadata["confidence"] == "high"
        assert result.metadata["status"] == "active"
        print(f"✓ 场景6 通过: 结果解析正确 id={result.id}, confidence={result.metadata['confidence']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Milvus expr 标量过滤集成测试（mock 端到端）")
    print("=" * 60)

    test_default_filter_passes_expr()
    test_none_filter_no_expr()
    test_confidence_filter_passes_expr()
    test_expr_failure_fallback()
    test_global_switch_disabled()
    test_query_results_parsed_correctly()

    print("=" * 60)
    print("全部 6 个场景测试通过 ✓")
    print("=" * 60)
