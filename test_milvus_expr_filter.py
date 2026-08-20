"""Milvus expr 标量过滤单元测试

测试 VectorSearchService._build_expr 方法，不依赖真实 Milvus 服务。
覆盖场景：默认过滤/不过滤/confidence过滤/组合过滤/全局开关关闭。
"""

import io
import sys
from pathlib import Path

# Windows 控制台 GBK 编码兼容
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.vector_search_service import VectorSearchService


def test_build_expr_default_active_filter():
    """场景1：默认 filter_status='active' → 生成 != 'deprecated' 兼容旧数据"""
    expr = VectorSearchService._build_expr("active", None)
    expected = "metadata['status'] != 'deprecated'"
    assert expr == expected, f"默认过滤失败: expected={expected}, got={expr}"
    print(f"✓ 场景1 通过: 默认 active 过滤 → {expr}")


def test_build_expr_no_filter():
    """场景2：filter_status=None → 不过滤返回 None"""
    expr = VectorSearchService._build_expr(None, None)
    assert expr is None, f"不过滤失败: expected=None, got={expr}"
    print(f"✓ 场景2 通过: None 过滤 → expr=None")


def test_build_expr_confidence_filter():
    """场景3：confidence 白名单过滤"""
    expr = VectorSearchService._build_expr(None, ["high", "medium"])
    expected = "metadata['confidence'] in ['high', 'medium']"
    assert expr == expected, f"confidence 过滤失败: expected={expected}, got={expr}"
    print(f"✓ 场景3 通过: confidence 过滤 → {expr}")


def test_build_expr_combined_filter():
    """场景4：status + confidence 组合过滤"""
    expr = VectorSearchService._build_expr("active", ["high", "medium"])
    expected = (
        "metadata['status'] != 'deprecated' and "
        "metadata['confidence'] in ['high', 'medium']"
    )
    assert expr == expected, f"组合过滤失败: expected={expected}, got={expr}"
    print(f"✓ 场景4 通过: 组合过滤 → {expr}")


def test_build_expr_custom_status():
    """场景5：自定义 status（非 active）→ 用 == 过滤"""
    expr = VectorSearchService._build_expr("pending", None)
    expected = "metadata['status'] == 'pending'"
    assert expr == expected, f"自定义 status 失败: expected={expected}, got={expr}"
    print(f"✓ 场景5 通过: 自定义 status=pending → {expr}")


def test_build_expr_empty_confidence():
    """场景6：空 confidence 列表 → 不过滤 confidence"""
    expr = VectorSearchService._build_expr("active", [])
    expected = "metadata['status'] != 'deprecated'"
    assert expr == expected, f"空 confidence 失败: expected={expected}, got={expr}"
    print(f"✓ 场景6 通过: 空 confidence 列表 → {expr}")


def test_global_switch_disabled(monkeypatch):
    """场景7：全局开关 milvus_expr_filter_enabled=False → expr 不生效

    这个测试模拟 config.milvus_expr_filter_enabled=False 时，
    search_similar_documents 内部应该走 expr=None 分支。
    """
    from app.config import config

    # 保存原值
    original = config.milvus_expr_filter_enabled
    try:
        config.milvus_expr_filter_enabled = False
        # 即使传了 filter_status="active"，_build_expr 仍然会生成 expr，
        # 但 search_similar_documents 内部会检查开关跳过 expr。
        # 这里验证 _build_expr 本身不受开关影响（开关在调用层控制）
        expr = VectorSearchService._build_expr("active", None)
        assert expr is not None, "_build_expr 本身不应受开关影响"
        print(f"✓ 场景7 通过: 全局开关关闭时 _build_expr 仍生成 expr={expr}，调用层负责跳过")
    finally:
        config.milvus_expr_filter_enabled = original


def test_expr_syntax_validity():
    """场景8：验证生成的 expr 语法符合 Milvus 规范

    Milvus JSON 字段过滤语法：
    - 字段访问：metadata['key']
    - 字符串值：'value'（单引号）
    - in 操作：in ['v1', 'v2']
    - 组合：and / or
    """
    test_cases = [
        ("active", None),
        ("active", ["high"]),
        ("active", ["high", "medium", "low"]),
        ("pending", None),
        (None, ["high"]),
        (None, None),
    ]

    for status, confidence in test_cases:
        expr = VectorSearchService._build_expr(status, confidence)
        if expr is None:
            continue
        # 验证语法：字符串值用单引号
        assert "'" in expr, f"expr 缺少单引号: {expr}"
        # 验证 metadata 字段访问
        assert "metadata[" in expr, f"expr 缺少 metadata 访问: {expr}"
        # 验证没有双引号（Milvus expr 用单引号）
        assert '"' not in expr, f"expr 不应含双引号: {expr}"

    print(f"✓ 场景8 通过: 所有 expr 语法符合 Milvus 规范")


if __name__ == "__main__":
    print("=" * 60)
    print("Milvus expr 标量过滤单元测试")
    print("=" * 60)

    test_build_expr_default_active_filter()
    test_build_expr_no_filter()
    test_build_expr_confidence_filter()
    test_build_expr_combined_filter()
    test_build_expr_custom_status()
    test_build_expr_empty_confidence()
    test_global_switch_disabled(None)
    test_expr_syntax_validity()

    print("=" * 60)
    print("全部 8 个场景测试通过 ✓")
    print("=" * 60)
