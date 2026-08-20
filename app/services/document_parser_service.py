"""文档解析服务模块 - 提取层：根据文件类型自动路由到对应提取器"""

from pathlib import Path
from typing import Callable, Dict

from loguru import logger

from app.config import config


class DocumentParserService:
    """文档解析服务 - 提取层

    根据文件扩展名自动路由到对应的提取器。
    所有提取器输出统一为纯文本字符串，供分片层处理。

    支持的文件类型:
        .txt  - 直接读取 UTF-8 文本
        .md   - 直接读取 UTF-8 文本（保留 Markdown 标记供分片层使用）
        .pdf  - pypdf 逐页提取文本 + pdfplumber 表格提取 + qwen-vl OCR 扫描页
        .docx - python-docx 逐段落提取文本（标题样式映射）+ 表格提取

    内嵌标记（由分片层解析后清除，不写入向量库）:
        [[PAGE:n]]    - PDF 页码标记（P0）
        [[TABLE]]...[[/TABLE]] - 表格块标记（P1，表格作为独立分片，不参与字符切分）
    """

    def __init__(self):
        self._parsers: Dict[str, Callable[[str], str]] = {
            ".txt": self._parse_text,
            ".md": self._parse_text,
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
        }
        logger.info(
            f"文档解析服务初始化完成, 支持类型: {', '.join(self._parsers.keys())}"
        )

    def parse(self, file_path: str) -> str:
        """
        解析文件，提取纯文本内容

        Args:
            file_path: 文件路径

        Returns:
            str: 提取的纯文本内容

        Raises:
            ValueError: 不支持的文件类型
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        parser = self._parsers.get(ext)
        if parser is None:
            raise ValueError(
                f"不支持的文件类型: {ext}，支持的类型: {', '.join(self._parsers.keys())}"
            )

        logger.info(f"开始解析文件: {path.name} (类型: {ext})")
        text = parser(file_path)
        logger.info(f"文件解析完成: {path.name} -> {len(text)} 字符")
        return text

    def _parse_text(self, file_path: str) -> str:
        """解析纯文本文件 (.txt, .md)"""
        return Path(file_path).read_text(encoding="utf-8")

    def _parse_pdf(self, file_path: str) -> str:
        """解析 PDF 文件

        每页按三通道处理：
        1. 扫描页（有效字符 < ocr_min_text_chars）→ PyMuPDF 渲染 + qwen-vl OCR 转写
        2. 有表格的页 → pdfplumber 提取表格转 Markdown（[[TABLE]] 标记），
           正文剔除表格区域文字避免重复入库
        3. 纯文字页 → pypdf 直接提取（快路径，行为同 P0）

        每页文本前插入 [[PAGE:n]] 页码标记，供分片层提取页码范围。
        """
        from pypdf import PdfReader

        import pdfplumber

        from app.services.ocr_service import ocr_service

        reader = PdfReader(file_path)
        parts = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                plumber_page = pdf.pages[i - 1] if i - 1 < len(pdf.pages) else None

                # 通道1：扫描页 → OCR（表格提取跳过：无文本层检测不到表格）
                if len(text) < config.ocr_min_text_chars and config.ocr_enabled:
                    ocr_text = ocr_service.ocr_page(file_path, i - 1)
                    if ocr_text:
                        parts.append(f"[[PAGE:{i}]]\n{ocr_text}")
                    # OCR 失败 → 跳过该页（等同旧版行为）
                    continue

                # 通道2：表格页 → 表格独立成块 + 正文剔除表格文字
                tables = self._find_pdf_tables(plumber_page, i)
                if tables:
                    body = self._pdf_text_without_tables(plumber_page, tables)
                    if body:
                        parts.append(f"[[PAGE:{i}]]\n{body}")
                    else:
                        # 纯表格页：保留页码标记，供表格块关联页码
                        parts.append(f"[[PAGE:{i}]]")
                    for table in tables:
                        md = self._rows_to_markdown(table.extract())
                        if md:
                            parts.append(f"[[TABLE]]\n{md}\n[[/TABLE]]")
                    continue

                # 通道3：纯文字页 → pypdf 快路径
                if text:
                    parts.append(f"[[PAGE:{i}]]\n{text}")

        return "\n".join(parts)

    @staticmethod
    def _find_pdf_tables(plumber_page, page_number: int) -> list:
        """检测 PDF 页中的表格（带异常降级：检测失败按无表格处理）"""
        if plumber_page is None or not config.pdf_table_extraction_enabled:
            return []
        try:
            return plumber_page.find_tables()
        except Exception as e:
            logger.warning(f"表格检测失败（本页按无表格处理）: 第{page_number}页, {e}")
            return []

    @staticmethod
    def _pdf_text_without_tables(plumber_page, tables) -> str:
        """提取页面正文（剔除所有表格区域内的文字，避免表格内容重复入库）"""
        region = plumber_page
        for table in tables:
            region = region.outside_bbox(table.bbox)
        return (region.extract_text() or "").strip()

    def _parse_docx(self, file_path: str) -> str:
        """解析 Word 文档 (.docx)

        按文档原始顺序遍历段落和表格：
        - 段落：标题样式映射为 Markdown 标记（P0 逻辑）
        - 表格：转为 Markdown 并用 [[TABLE]] 标记包裹，附带当前章节标题作为上下文
          （表格脱离标题语义会失真，检索时对不上）
        """
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(file_path)
        parts = []
        current_heading = ""  # 表格所属章节上下文

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, doc)
                text = paragraph.text.strip()
                if not text:
                    continue
                style_name = paragraph.style.name if paragraph.style is not None else ""
                prefix = self._docx_heading_prefix(style_name or "")
                if prefix:
                    current_heading = text  # 更新章节上下文
                parts.append(f"{prefix}{text}" if prefix else text)

            elif isinstance(child, CT_Tbl):
                table = Table(child, doc)
                md = self._docx_table_to_markdown(table)
                if md:
                    heading_ctx = f"（所属章节: {current_heading}）\n\n" if current_heading else ""
                    parts.append(f"[[TABLE]]\n{heading_ctx}{md}\n[[/TABLE]]")

        return "\n".join(parts)

    @staticmethod
    def _docx_table_to_markdown(table) -> str:
        """docx 表格对象 → Markdown 表格文本"""
        rows = []
        for row in table.rows:
            cells = [DocumentParserService._clean_cell(cell.text) for cell in row.cells]
            rows.append(cells)
        return DocumentParserService._rows_to_markdown(rows)

    @staticmethod
    def _clean_cell(text: str) -> str:
        """清理表格单元格文本（转义管道符、合并换行）"""
        return (text or "").replace("|", "\\|").replace("\n", " ").strip()

    @staticmethod
    def _rows_to_markdown(rows) -> str:
        """二维数据 → Markdown 表格文本

        过滤误检：<2 行或 <2 列的"表格"大概率是分隔线/装饰框，返回空串。
        """
        # 规整化：None → 空串；管道符转义防破坏表格结构；单元格内换行合并为空格
        def _clean(cell) -> str:
            if cell is None:
                return ""
            return str(cell).strip().replace("|", "\\|").replace("\n", " ")

        rows = [[_clean(c) for c in r] for r in rows]
        rows = [r for r in rows if any(c for c in r)]

        if len(rows) < 2 or max(len(r) for r in rows) < 2:
            return ""

        # 补齐列数（行列不齐的表格）
        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]

        header, data_rows = rows[0], rows[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "---|" * n_cols,
        ]
        for r in data_rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _docx_heading_prefix(style_name: str) -> str:
        """将 docx 段落样式名映射为 Markdown 标题前缀（非标题样式返回空串）

        兼容英文内置样式名（Heading 1）和中文样式名（标题 1）。
        只映射 H1/H2 两级，与 Markdown 分片策略对齐，避免过度碎片化。
        """
        if style_name in ("Heading 1", "标题 1"):
            return "# "
        if style_name in ("Heading 2", "标题 2"):
            return "## "
        return ""


# 全局单例
document_parser_service = DocumentParserService()
