# 多类型文档解析与向量化索引实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 支持 TXT/Markdown/PDF/Word 四种文件类型的自动解析、文本提取、分片和向量化索引

**Architecture:** 新增 `DocumentParserService` 作为提取层（注册表模式路由到不同提取器），与现有分片层解耦。PDF 用 `pypdf`，Word 用 `python-docx`，均无需系统级依赖。

**Tech Stack:** Python, pypdf>=5.0.0, python-docx>=1.1.0, LangChain

---

### Task 1: 添加依赖 + 新建提取层 document_parser_service.py

**Files:**
- Modify: `pyproject.toml`
- Create: `app/services/document_parser_service.py`

- [ ] **Step 1: pyproject.toml 新增依赖**

在 `dependencies` 列表中增加：

```toml
    "pypdf>=5.0.0",
    "python-docx>=1.1.0",
```

- [ ] **Step 2: 创建 document_parser_service.py**

```python
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
```

---

### Task 2: 修改 vector_index_service.py — 接入提取层

**Files:**
- Modify: `app/services/vector_index_service.py`

- [ ] **Step 1: 增加导入 + 修改 index_single_file**

```python
"""向量索引服务模块"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services.document_parser_service import document_parser_service  # 新增
from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


class VectorIndexService:
    """向量索引服务 - 负责解析文件、分割文档、生成向量并存储到 Milvus"""

    # ... IndexingResult, index_directory等方法不变 ...

    def index_single_file(self, file_path: str):
        """索引单个文件

        Args:
            file_path: 文件路径

        Raises:
            ValueError: 文件不存在时抛出
            RuntimeError: 索引失败时抛出
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")

        try:
            # 1. 使用提取层解析文件内容（自动根据扩展名路由）
            content = document_parser_service.parse(str(path))
            logger.info(f"解析文件: {path}, 内容长度: {len(content)} 字符")

            # 2. 删除该文件的旧数据（如果存在）
            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)

            # 3. 使用分片层分割文档
            documents = document_splitter_service.split_document(content, normalized_path)
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            # 4. 添加文档到向量存储
            if documents:
                vector_store_manager.add_documents(documents)
                logger.info(f"文件索引完成: {file_path}, 共 {len(documents)} 个分片")
            else:
                logger.warning(f"文件内容为空或无法分割: {file_path}")

        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"索引文件失败: {e}") from e
```

---

### Task 3: 修改 file.py — 扩展支持的文件类型

**Files:**
- Modify: `app/api/file.py`

- [ ] **Step 1: 修改 ALLOWED_EXTENSIONS**

```python
# 支持的文件类型
ALLOWED_EXTENSIONS = ["txt", "md", "pdf", "docx"]
```

---

### Task 4: 编写文档

**Files:**
- Create: `docs/更新日志—多类型文档解析.md`

内容包含：
- 架构图
- 每种文件类型的分片方式
- 每处改动的代码对比
- 如何新增一个文件类型的步骤
