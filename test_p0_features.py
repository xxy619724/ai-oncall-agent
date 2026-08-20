"""P0 优化项单元测试

覆盖三项纯逻辑改动（不依赖 Milvus / LLM 等外部服务）：
1. P0-1 Word 标题样式 → Markdown 前缀映射
2. P0-2 PDF 页码标记提取与清除（_extract_page_metadata）
3. P0-1/P0-2 分片路由（docx → Markdown 链路，pdf → 页码提取）

运行: .venv\\Scripts\\python.exe test_p0_features.py
"""

import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document


def test_docx_heading_prefix():
    """P0-1：docx 样式名 → Markdown 标题前缀映射"""
    from app.services.document_parser_service import DocumentParserService

    prefix = DocumentParserService._docx_heading_prefix

    assert prefix("Heading 1") == "# ", "英文 Heading 1 应映射为 #"
    assert prefix("Heading 2") == "## ", "英文 Heading 2 应映射为 ##"
    assert prefix("标题 1") == "# ", "中文 标题 1 应映射为 #"
    assert prefix("标题 2") == "## ", "中文 标题 2 应映射为 ##"
    assert prefix("Normal") == "", "正文样式应返回空串"
    assert prefix("Heading 3") == "", "H3 不映射（对齐 Markdown 只切 H1/H2 策略）"
    assert prefix("") == "", "空样式名应返回空串"

    print("✓ P0-1 docx 标题样式映射")


def test_extract_page_metadata_basic():
    """P0-2：完整页码标记 → 页码范围 + 标记清除"""
    from app.services.document_splitter_service import document_splitter_service

    doc = Document(page_content="[[PAGE:3]]\n第一页内容。\n[[PAGE:5]]\n跨页内容。")
    result = document_splitter_service._extract_page_metadata([doc])

    assert len(result) == 1
    assert result[0].metadata["_page_start"] == 3
    assert result[0].metadata["_page_end"] == 5
    assert "[[PAGE" not in result[0].page_content, "标记应被清除"
    assert "第一页内容" in result[0].page_content
    assert "跨页内容" in result[0].page_content

    print("✓ P0-2 页码范围提取与标记清除")


def test_extract_page_metadata_partial():
    """P0-2：被切分器截断的半截标记也能清除"""
    from app.services.document_splitter_service import document_splitter_service

    doc = Document(page_content="正文内容。[[PAGE:1\n后续文本。")
    result = document_splitter_service._extract_page_metadata([doc])

    assert len(result) == 1
    assert "[[PAGE" not in result[0].page_content, "半截标记应被兜底清除"
    assert "_page_start" not in result[0].metadata, "半截标记不应产生页码元数据"

    print("✓ P0-2 半截标记兜底清除")


def test_extract_page_metadata_drop_empty():
    """P0-2：纯标记无正文的分片被丢弃"""
    from app.services.document_splitter_service import document_splitter_service

    doc = Document(page_content="[[PAGE:7]]")
    result = document_splitter_service._extract_page_metadata([doc])

    assert result == [], "纯标记分片应被丢弃"

    print("✓ P0-2 纯标记分片丢弃")


def test_split_document_pdf_routing():
    """P0-2：PDF 分片端到端 → chunk 携带页码元数据且正文无标记"""
    from app.services.document_splitter_service import document_splitter_service

    # 模拟 _parse_pdf 输出（每页前有 [[PAGE:n]] 标记）
    content = (
        "[[PAGE:1]]\nCPU 告警处理：检查进程占用，使用 top 命令定位异常进程。\n"
        "[[PAGE:2]]\n内存告警处理：检查内存泄漏，使用 free 命令确认使用率。\n"
    )
    docs = document_splitter_service.split_document(content, "uploads/test.pdf")

    assert docs, "应产出分片"
    for d in docs:
        assert "[[PAGE" not in d.page_content, f"分片正文不应残留标记: {d.page_content[:50]}"
    # 至少一个分片带页码元数据
    with_page = [d for d in docs if "_page_start" in d.metadata]
    assert with_page, "应有分片携带 _page_start 元数据"
    for d in with_page:
        assert d.metadata["_page_end"] >= d.metadata["_page_start"]

    print(f"✓ P0-2 PDF 路由端到端（{len(docs)} 个分片，{len(with_page)} 个带页码）")


def test_split_document_docx_routing():
    """P0-1：docx 路由到 Markdown 标题切分链路 + 扩展名修正"""
    from app.services.document_splitter_service import document_splitter_service

    # 模拟 _parse_docx 输出（标题样式已映射为 # / ## 标记）
    # 注意：各 h2 小节内容需超过 300 字符，避免被小片段合并成单块（合并只保留首个分片的元数据）
    filler = "排查时先确认告警时间线，再核对监控指标基线，避免误判。" * 15  # 约 405 字符
    content = (
        "# CPU 高负载处理\n\n总体思路：先定位进程，再确认阈值。\n\n"
        f"## 步骤一：定位进程\n\n{filler}\n\n"
        f"## 步骤二：确认阈值\n\n{filler}\n"
    )
    docs = document_splitter_service.split_document(content, "uploads/test.docx")

    assert docs, "应产出分片"
    assert any(d.metadata.get("h1") == "CPU 高负载处理" for d in docs), "应按 H1 标题切分并写入 h1 元数据"
    h2_docs = [d for d in docs if d.metadata.get("h2")]
    assert h2_docs, "应有分片携带 h2 标题元数据"
    assert any(d.metadata.get("h2") == "步骤一：定位进程" for d in h2_docs), "h2 元数据应包含章节标题"
    for d in docs:
        assert d.metadata["_extension"] == ".docx", "扩展名应为 .docx 而非硬编码 .md"
        assert "_page_start" not in d.metadata, "docx 不应误加页码元数据"

    print(f"✓ P0-1 docx 路由到 Markdown 链路（{len(docs)} 个分片，h2 分片 {len(h2_docs)} 个）")


def main():
    print("=" * 60)
    print("P0 优化项单元测试")
    print("=" * 60)

    test_docx_heading_prefix()
    test_extract_page_metadata_basic()
    test_extract_page_metadata_partial()
    test_extract_page_metadata_drop_empty()
    test_split_document_pdf_routing()
    test_split_document_docx_routing()

    print("\n" + "=" * 60)
    print("全部通过 ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
