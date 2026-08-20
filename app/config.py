"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Any, Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # CORS 配置
    # 逗号分隔的来源白名单。留空 = 不放行任何跨域来源（内置前端与后端同源，无需 CORS）。
    # 注意：通配符 "*" 与 allow_credentials=True 组合会被浏览器拒绝，故此处不支持 "*" + 凭证。
    cors_allow_origins: str = ""

    # /api/index_directory 目录白名单（逗号分隔，相对项目根目录）
    # 该端点会把目录下的文件读入向量库，必须限制范围，否则可被用于读取任意路径。
    index_allowed_dirs: str = "uploads,aiops-docs"

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 国内站点（默认勿用国际站 dashscope-intl）
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # Milvus 标量过滤配置（expr 表达式）
    milvus_expr_filter_enabled: bool = True  # 总开关：False 时所有搜索不带 expr 降级到无过滤
    milvus_default_status_filter: str = "active"  # RAG 检索默认只召回 status=active 的经验

    # RAG 配置
    rag_top_k: int = 10  # 召回阶段检索数量（供重排筛选）
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 聊天链路多模态配置（Qwen-VL 视觉模型，支持图片理解）
    chat_model: str = "qwen-vl-max"  # 聊天主模型（多模态，纯文本请求也兼容）
    chat_image_max_base64_size: int = 4 * 1024 * 1024  # 单图 base64 字符串长度上限（约 3MB 原图）

    # 重排（Rerank）配置
    rag_rerank_top_k: int = 3  # 重排后保留的文档数
    rag_rerank_model: str = "qwen3-rerank"  # 百炼 rerank 模型
    rag_relevance_score_threshold: float = 0.3  # rerank 分数阈值，低于此值过滤低质量文档

    # 上下文压缩配置
    rag_context_window_size: int = 131072  # qwen-max 上下文窗口（128K tokens）
    rag_compression_threshold: float = 0.7  # 触发压缩的 token 占比阈值
    rag_keep_recent_rounds: int = 5  # 保留最近几轮完整原文（滑动窗口）
    rag_compression_model: str = "qwen-max"  # 压缩用的模型

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # OCR 扫描件配置（P1：qwen-vl 云端 OCR）
    ocr_enabled: bool = True                  # 总开关：False 时扫描页直接跳过（等同旧行为）
    ocr_model: str = "qwen-vl-max"            # OCR 用的多模态模型
    ocr_timeout: float = 60.0                 # OCR 单页请求超时（秒）
    ocr_min_text_chars: int = 50              # 单页有效字符低于此值判定为扫描页

    # PDF 表格提取配置（P1：pdfplumber 独立分片）
    pdf_table_extraction_enabled: bool = True  # 总开关：False 时不提取表格（等同旧行为）

    # 语义缓存配置（P1：完整回答级缓存）
    semantic_cache_enabled: bool = True        # 总开关：False 时完全旁路缓存
    semantic_cache_threshold: float = 0.95     # 命中相似度阈值（COSINE，越高越保守）
    semantic_cache_ttl_hours: int = 24         # 缓存 TTL（小时），过期自动失效

    # MCP 服务配置（transport: stdio | sse | streamable-http）
    # 腾讯云托管 MCP 的 URL 通常含 /sse/，需使用 sse；本地 FastMCP 使用 streamable-http
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # Prometheus
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    # 对话记忆持久化配置
    sqlite_db_path: str = "./data/chat.db"
    redis_url: str = "redis://localhost:6379/0"
    redis_summary_ttl: int = 604800  # 7天（秒）

    # SQLite Checkpoint 自动清理配置
    sqlite_checkpoint_max_age_days: int = 7      # checkpoint 保留天数（超过则清理全量对话）
    sqlite_cleanup_interval_hours: int = 24      # 定时清理间隔（小时）
    sqlite_cleanup_batch_size: int = 100          # 每批清理会话数（避免锁库）

    # 记忆写入门控配置
    memory_dedup_threshold: float = 0.95     # 查重阈值（相似度≥此值判定重复，跳过写入）
    memory_conflict_threshold: float = 0.80  # 冲突检测阈值（相似度≥此值进入冲突检测）
    memory_default_ttl_days: int = 90        # 经验默认 TTL（天，超时后可软删除）

    # 经验记忆 TTL 定时清理配置
    memory_ttl_cleanup_interval_hours: int = 24   # TTL 检查间隔（小时）
    memory_ttl_cleanup_batch_size: int = 100      # 每批标记 deprecated 数量（避免锁库）

    # 可观测体系配置（Trace/Span/Metric）
    observability_enabled: bool = True            # 总开关：False 时所有埋点零开销直通
    observability_db_path: str = "./data/observability.db"  # 独立于 checkpoint 库
    observability_span_input_max_len: int = 500   # Span 输入摘要截断长度（字符）
    observability_span_output_max_len: int = 500  # Span 输出摘要截断长度（字符）
    observability_trace_input_max_len: int = 500  # Trace 输入截断长度（字符）

    # 异步任务系统配置（阶段一：最小可用异步任务系统）
    task_db_path: str = "./data/tasks.db"         # 任务状态持久化 SQLite（独立库）
    task_queue_maxsize: int = 100                  # 任务队列容量（满时返回 503）
    task_timeout_seconds: int = 300                # 任务执行硬超时（秒），超时标记 failed
    task_event_buffer_size: int = 200              # 单任务事件缓冲区大小（内存）
    task_worker_concurrency: int = 1               # Worker 并发数（阶段一固定为 1）
    # 任务结束后事件在内存中的保留时长（秒）。留出这段时间给仍在读取的 SSE 消费者，
    # 到期后清理，避免 event_store 随任务数无限增长。
    task_event_retention_seconds: int = 300

    # LLM 并发控制（防止 API 限流 + httpx 连接池耗尽）
    llm_concurrency_limit: int = 3                 # 同时调用 LLM 的最大数量

    @property
    def cors_origins_list(self) -> List[str]:
        """解析 CORS 白名单为列表（空配置返回空列表，即不放行跨域）"""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def index_allowed_dirs_list(self) -> List[str]:
        """解析索引目录白名单为列表"""
        return [d.strip() for d in self.index_allowed_dirs.split(",") if d.strip()]

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
