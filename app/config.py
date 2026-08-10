"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
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

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 国内站点（默认勿用国际站 dashscope-intl）
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 10  # 召回阶段检索数量（供重排筛选）
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 重排（Rerank）配置
    rag_rerank_top_k: int = 3  # 重排后保留的文档数
    rag_rerank_model: str = "qwen3-rerank"  # 百炼 rerank 模型

    # 上下文压缩配置
    rag_context_window_size: int = 131072  # qwen-max 上下文窗口（128K tokens）
    rag_compression_threshold: float = 0.7  # 触发压缩的 token 占比阈值
    rag_keep_recent_rounds: int = 5  # 保留最近几轮完整原文（滑动窗口）
    rag_compression_model: str = "qwen-max"  # 压缩用的模型

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

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

    # 可观测体系配置（Trace/Span/Metric）
    observability_enabled: bool = True            # 总开关：False 时所有埋点零开销直通
    observability_db_path: str = "./data/observability.db"  # 独立于 checkpoint 库
    observability_span_input_max_len: int = 500   # Span 输入摘要截断长度（字符）
    observability_span_output_max_len: int = 500  # Span 输出摘要截断长度（字符）
    observability_trace_input_max_len: int = 500  # Trace 输入截断长度（字符）

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
