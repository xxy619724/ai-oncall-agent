"""文档解析服务模块 - 提取层：根据文件类型自动路由到对应提取器"""

from pathlib import Path
from typing import Callable, Dict

from loguru import logger


class DocumentParserService:
    """文档解析服务 - 提取层

    根据文件扩展名自动路由到对应的提取器。
    所有提取器输出统一为纯文本字符串，供分片层处理。

    支持的文件类型:
        .txt  - 直接读取 UTF-8 文本
        .md   - 直接读取 UTF-8 文本（保留 Markdown 标记供分片层使用）
        .pdf  - 使用 pypdf 逐页提取文本
        .docx - 使用 python-docx 逐段落提取文本
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
        """解析 PDF 文件"""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                pages_text.append(text)

        return "\n".join(pages_text)

    def _parse_docx(self, file_path: str) -> str:
        """解析 Word 文档 (.docx)"""
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)


# 全局单例
document_parser_service = DocumentParserService()
