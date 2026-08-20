"""文档分割服务模块 - 基于 LangChain 的智能文档分割"""

import re
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config

# PDF 页码标记：提取层在每页前插入 [[PAGE:n]]，分片后提取页码范围并清除标记
_PAGE_MARKER_RE = re.compile(r"\[\[PAGE:(\d+)\]\]")
# 兜底清理被切分器截断的半截标记（如 "[[PAGE:12" / "[[PAGE:"）
_PAGE_MARKER_ANY_RE = re.compile(r"\[\[PAGE:\d*\]*")
# 表格块标记：提取层用 [[TABLE]]...[[/TABLE]] 包裹表格 Markdown（P1）
_TABLE_BLOCK_RE = re.compile(r"\[\[TABLE\]\]\n(.*?)\n?\[\[/TABLE\]\]", re.DOTALL)
# 表格分片字符上限（Milvus content 字段上限 8000，留余量）
_MAX_TABLE_CHUNK_CHARS = 7000


class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        """初始化文档分割服务"""
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # Markdown 标题分割器 (只按一级和二级标题分割，减少分片数)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # 不再按三级标题分割，避免过度碎片化
            ],
            strip_headers=False,  # 保留标题在内容中
        )

        # 递归字符分割器 (用于二次分割，使用更大的chunk_size)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # 加倍chunk_size，减少分片数
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # 第一阶段: 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段: 按大小进一步分割
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 第三阶段: 合并太小的分片 (< 300字符)
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据
            for doc in final_docs:
                doc.metadata["_source"] = file_path
                # 使用实际扩展名（docx 复用此链路时不能被误标为 .md）
                doc.metadata["_extension"] = Path(file_path).suffix or ".md"
                doc.metadata["_file_name"] = Path(file_path).name

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        # P1：先抽离表格块（表格作为独立分片，不参与字符切分，避免被切断）
        table_blocks: List[Dict] = []
        if file_path.endswith((".pdf", ".docx")):
            content, table_blocks = self._extract_table_blocks(content)

        if file_path.endswith(".md"):
            docs = self.split_markdown(content, file_path)
        elif file_path.endswith(".docx"):
            # docx 提取层已将标题样式映射为 #/## 标记，复用 Markdown 标题切分链路
            docs = self.split_markdown(content, file_path)
        else:
            docs = self.split_text(content, file_path)

        # PDF：提取页码标记 → metadata 页码范围，并清除标记正文
        # （仅对 .pdf 生效，避免误伤其他类型文档中的同形文本）
        if file_path.endswith(".pdf"):
            docs = self._extract_page_metadata(docs)

        # P1：表格块追加为独立分片（带 _type=table 元数据，超长按行切分）
        if table_blocks:
            docs.extend(self._build_table_chunks(table_blocks, file_path))

        return docs

    def _extract_table_blocks(self, content: str):
        """
        从文本流中抽离表格块标记（[[TABLE]]...[[/TABLE]]）

        表格块在字符切分之前抽出（避免被 1600 字符边界切断），
        PDF 表格块通过其前方最近的 [[PAGE:n]] 标记关联页码。

        Args:
            content: 提取层输出的文本（含表格块标记）

        Returns:
            tuple[str, List[dict]]: (去除表格块后的文本, 表格块列表)
                表格块 dict: {markdown: 表格Markdown, page: 页码或None}
        """
        blocks: List[Dict] = []

        def _pull(match) -> str:
            # 关联页码：表格块前方最近一次出现的 [[PAGE:n]]
            pages_before = _PAGE_MARKER_RE.findall(content[: match.start()])
            page = int(pages_before[-1]) if pages_before else None
            blocks.append({"markdown": match.group(1).strip(), "page": page})
            return "\n"  # 从文本流中移除表格块

        new_content = _TABLE_BLOCK_RE.sub(_pull, content)
        if blocks:
            logger.info(f"抽离表格块: {len(blocks)} 个")
        return new_content, blocks

    def _build_table_chunks(
        self, blocks: List[Dict], file_path: str
    ) -> List[Document]:
        """
        表格块列表 → 独立文档分片

        每个表格一个分片（不参与字符切分）；超过 Milvus content 上限时
        按行切分，每个分片重复表头并标记 _table_part 序号。

        Args:
            blocks: _extract_table_blocks 抽离的表格块列表
            file_path: 文件路径（元数据）

        Returns:
            List[Document]: 表格分片列表
        """
        chunks: List[Document] = []
        for block in blocks:
            markdown = block["markdown"]
            base_meta = {
                "_source": file_path,
                "_extension": Path(file_path).suffix,
                "_file_name": Path(file_path).name,
                "_type": "table",
            }
            if block.get("page") is not None:
                base_meta["_page_start"] = block["page"]
                base_meta["_page_end"] = block["page"]

            if len(markdown) <= _MAX_TABLE_CHUNK_CHARS:
                chunks.append(Document(page_content=markdown, metadata=dict(base_meta)))
                continue

            # 超长表格：按行切分，每个分片重复表头（表头 + 分隔行在前两行）
            lines = markdown.split("\n")
            header = lines[:2]
            rows = lines[2:]
            parts: List[str] = []
            current = list(header)
            for row in rows:
                candidate_len = len("\n".join(current)) + len(row) + 1
                if (
                    candidate_len > _MAX_TABLE_CHUNK_CHARS
                    and len(current) > len(header)
                ):
                    parts.append("\n".join(current))
                    current = list(header)
                current.append(row)
            if len(current) > len(header):
                parts.append("\n".join(current))

            total = len(parts)
            for idx, part in enumerate(parts, 1):
                meta = dict(base_meta)
                meta["_table_part"] = f"{idx}/{total}"
                chunks.append(Document(page_content=part, metadata=meta))

            logger.warning(
                f"超长表格按行切分: {file_path}, {len(markdown)} 字符 -> {total} 片"
            )

        return chunks

    def _extract_page_metadata(self, documents: List[Document]) -> List[Document]:
        """
        从分片中提取 PDF 页码标记

        每个包含 [[PAGE:n]] 标记的分片，记录 _page_start/_page_end 到 metadata，
        并从正文中清除标记（含被切分器截断的半截标记）。

        Args:
            documents: 文档分片列表

        Returns:
            List[Document]: 处理后的分片列表（纯标记无正文的分片会被丢弃）
        """
        result = []
        for doc in documents:
            content = doc.page_content
            pages = [int(m) for m in _PAGE_MARKER_RE.findall(content)]
            if pages:
                doc.metadata["_page_start"] = min(pages)
                doc.metadata["_page_end"] = max(pages)

            cleaned = _PAGE_MARKER_ANY_RE.sub("", content).strip()
            if not cleaned:
                # 极端情况：分片只含页码标记无正文，丢弃避免入库空内容
                continue
            doc.page_content = cleaned
            result.append(doc)

        return result

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """
        合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小 (字符数)

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                # 当前文档太小且合并后不会太大，则合并
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs


# 全局单例
document_splitter_service = DocumentSplitterService()
