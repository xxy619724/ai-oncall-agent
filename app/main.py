"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops, task
from app.core.milvus_client import milvus_manager
from app.observability import observability_store
from app.services.task_service import init_task_service, cleanup_task_service
from app.services.task_worker import start_task_worker, stop_task_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # 全站无鉴权，绑到非本机地址等于把接口开放给同网段：
    # /api/upload 可灌向量库、/api/chat 可耗尽 API 额度。显式告警而非静默放行。
    if config.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            f"⚠️ 正在监听 {config.host}（非本机地址），而本服务尚未接入任何鉴权。"
            f"同网段任何人都可调用 /api/upload、/api/index_directory、/api/chat。"
            f"仅在已配置反向代理鉴权或可信网络内使用。"
        )


    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")
    
    # MemoryService Redis 连接状态（实际初始化在 RagAgentService._initialize_agent 中）
    from app.services.memory_service import get_memory_service
    mem_service = get_memory_service()
    if mem_service and mem_service._redis_available:
        logger.info("✅ Redis 摘要层已就绪")
    else:
        logger.warning("⚠️ Redis 摘要层未就绪（降级为纯 SQLite 模式）")

    # 初始化可观测数据存储（Trace/Span/Metric）
    logger.info("📊 正在初始化可观测数据存储...")
    await observability_store.initialize()
    if observability_store.available:
        logger.info("✅ 可观测数据存储已就绪（Trace/Span/Metric）")
    else:
        logger.warning("⚠️ 可观测数据存储未就绪（埋点将降级为零开销直通）")

    # 初始化异步任务系统（TaskService + TaskWorker）
    logger.info("📋 正在初始化异步任务系统...")
    await init_task_service()
    await start_task_worker()
    logger.info("✅ 异步任务系统已就绪（POST /api/tasks 提交，GET /api/tasks/{id} 查询）")

    logger.info("=" * 60)

    yield

    # 关闭时执行
    logger.info("🔌 正在关闭服务...")

    # 停止异步任务 Worker
    await stop_task_worker()
    await cleanup_task_service()

    # 清理 RAG Agent 资源（SQLite + Redis）
    from app.services.rag_agent_service import rag_agent_service
    await rag_agent_service.cleanup()

    # 关闭 AIOps 工作流的 SQLite 连接（checkpoint + 经验表）
    from app.services.aiops_service import aiops_service
    await aiops_service.cleanup()

    # 关闭可观测数据存储连接（须在最后，前面的清理仍可能产生埋点）
    await observability_store.cleanup()

    # 关闭 Milvus
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
# 内置前端由本服务同源提供（/static），同源请求不经过 CORS，因此默认空白名单即可。
# 需要跨域访问时通过 CORS_ALLOW_ORIGINS 显式列出来源，不使用 "*"：
# "*" 与 allow_credentials=True 的组合会被浏览器拒绝，且等同于对任意站点开放接口。
_cors_origins = config.cors_origins_list
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    logger.info(f"CORS 已启用，允许来源: {_cors_origins}")
else:
    logger.info("CORS 未配置任何来源（仅同源访问）")

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])
app.include_router(task.router, prefix="/api", tags=["异步任务系统"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
