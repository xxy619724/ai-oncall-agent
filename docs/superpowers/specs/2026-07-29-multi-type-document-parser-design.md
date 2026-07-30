# 多类型文档解析与向量化索引设计

**目标：** 支持 TXT/Markdown/PDF/Word 四种文件类型的自动解析、文本提取、分片和向量化索引，采用提取层与分片层解耦架构。

## 当前问题

### 现有实现

```
upload API (file.py)     → 只接受 .txt / .md
vector_index_service.py  → path.read_text() 硬编码读取
document_splitter.py     → .md 按标题分割，其余按字符分割
```

### 三个问题

1. **文件格式覆盖不足**：只支持 `.txt` 和 `.md`，PDF 和 Word 不支持
2. **提取与分片耦合**：`vector_index_service.py` 直接调用 `path.read_text()`，新增格式必须在索引层改代码
3. **不易扩展**：新增一种文件类型需要改动 3 个文件（API 验证、索引读取、分割路由）

## 架构：提取层与分片层解耦

### 分层设计

```
┌─────────────────────────────────────────────────────┐
│                   API 层 (file.py)                    │
│            ALLOWED_EXTENSIONS = [txt, md, pdf, docx]  │
└─────────────────────┬───────────────────────────────┘
                      │ 上传文件
                      ▼
┌─────────────────────────────────────────────────────┐
│               提取层 (document_parser_service.py)     │  ← 新增
│                                                     │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐           │
│   │ TXT  │  │  MD  │  │ PDF  │  │ DOCX │           │
│   │读文本 │  │读文本 │  │pypdf │  │py-docx│           │
│   └──────┘  └──────┘  └──────┘  └──────┘           │
│                                                     │
│   输出：统一纯文本字符串                                │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              分片层 (document_splitter_service.py)    │  ← 微调
│                                                     │
│   .md → MarkdownHeader + RecursiveCharacter Splitter │
│   其余 → RecursiveCharacter Splitter                 │
│                                                     │
│   输出：List[Document]                                │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│            向量存储 (vector_store_manager.py)         │
│                  → Milvus                            │
└─────────────────────────────────────────────────────┘
```

### 设计原则

- **提取器注册表模式**：`{".txt": fn_txt, ".pdf": fn_pdf, ...}`，新增类型只需加一条映射
- **单一路径**：所有文件类型最终都输出纯文本，分片层统一处理
- **对扩展开放**：新增文件类型 = 写一个提取函数 + 注册到字典

## 文件修改清单

### 1. 新建 `app/services/document_parser_service.py`

提取层核心，职责：根据文件扩展名路由到对应提取器。

```python
class DocumentParserService:
    _parsers = {
        ".txt":  _parse_text,    # 直接 read_text(utf-8)
        ".md":   _parse_text,    # 直接 read_text(utf-8)，保留 Markdown 语法
        ".pdf":  _parse_pdf,     # pypdf.PdfReader 逐页提取
        ".docx": _parse_docx,    # docx.Document 逐段提取
    }

    def parse(self, file_path: str) -> str:
        """提取文本，未注册的类型抛 ValueError"""
```

#### 各提取器细节

**`.txt` / `.md`：**
```python
Path(file_path).read_text(encoding="utf-8")
```
Markdown 直接读为文本，保留标题标记（#、##），供分片层的 `MarkdownHeaderTextSplitter` 使用。

**`.pdf`（使用 `pypdf`）：**
```python
reader = PdfReader(file_path)
text = "\n".join(page.extract_text() for page in reader.pages)
```
- 逐页提取文本，用换行拼接
- 纯文本 PDF 可正常提取，扫描件不支持
- `pypdf` 是纯 Python 实现，无系统依赖

**`.docx`（使用 `python-docx`）：**
```python
doc = Document(file_path)
text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
```
- 逐段落提取文本，跳过空段落
- 图片和表格内容不提取（RAG 场景主要需求是文本）
- `python-docx` 是纯 Python 实现，无系统依赖

### 2. 修改 `app/services/document_splitter_service.py`

当前 `split_document()` 已按扩展名路由：
```python
def split_document(self, content: str, file_path: str = "") -> List[Document]:
    if file_path.endswith(".md"):
        return self.split_markdown(content, file_path)
    else:
        return self.split_text(content, file_path)
```

无需改动。`.pdf` 和 `.docx` 提取出的纯文本会走 `split_text` 分支。

唯一增强：在 `split_text` 的 metadata 中添加 `_extension` 字段（已存在）。

### 3. 修改 `app/services/vector_index_service.py`

`index_single_file` 方法：

**改前：**
```python
content = path.read_text(encoding="utf-8")
documents = document_splitter_service.split_document(content, normalized_path)
```

**改后：**
```python
content = document_parser_service.parse(str(path))
documents = document_splitter_service.split_document(content, normalized_path)
```

### 4. 修改 `app/api/file.py`

**改前：**
```python
ALLOWED_EXTENSIONS = ["txt", "md"]
```

**改后：**
```python
ALLOWED_EXTENSIONS = ["txt", "md", "pdf", "docx"]
```

### 5. 修改 `pyproject.toml`

```toml
dependencies = [
    ...
    "pypdf>=5.0.0",
    "python-docx>=1.1.0",
]
```

### 6. 新建 `docs/更新日志—多类型文档解析.md`

包含：
- 架构图（提取层/分片层）
- 每种文件类型的分片方式说明
- 每处改动的代码对比

## 各文件类型分片方式一览

| 文件类型 | 提取工具 | 提取方式 | 分片策略 |
|---|---|---|---|
| `.txt` | UTF-8 直接读取 | 读为纯文本 | RecursiveCharacterTextSplitter(1600/100) |
| `.md` | UTF-8 直接读取 | 读为纯文本（保留 #/## 标记） | MarkdownHeaderTextSplitter(h1/h2) → RecursiveCharacterTextSplitter(1600/100) → 合并小片段 |
| `.pdf` | `pypdf.PdfReader` | 逐页提取文本，换行拼接 | RecursiveCharacterTextSplitter(1600/100) |
| `.docx` | `python-docx.Document` | 逐段落提取文本，换行拼接 | RecursiveCharacterTextSplitter(1600/100) |

## 不变的部分

- `app/services/vector_store_manager.py` — 无需修改
- `app/core/milvus_client.py` — 无需修改
- `app/services/vector_embedding_service.py` — 无需修改
- `app/services/rerank_service.py` — 无需修改
- 前端 `static/` — 无需修改
