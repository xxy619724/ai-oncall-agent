"""LLM 调用并发控制（全局 Semaphore）

作用：
- 限制同时调用 LLM API 的数量
- 防止 LLM API 并发限流（429 Too Many Requests）
- 防止 httpx 连接池耗尽导致新请求无法建立连接

用法：
    from app.services.llm_semaphore import get_llm_semaphore

    async with get_llm_semaphore():
        response = await llm.ainvoke(messages)
"""

import asyncio
from typing import Optional

from loguru import logger

from app.config import config


_semaphore: Optional[asyncio.Semaphore] = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """获取全局 LLM 并发控制 Semaphore（懒加载）

    首次调用时创建 Semaphore，后续复用同一实例。
    必须在事件循环内调用（asyncio.Semaphore 要求）。

    Returns:
        asyncio.Semaphore 实例
    """
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.llm_concurrency_limit)
        logger.info(
            f"LLM 并发控制已初始化: max_concurrency={config.llm_concurrency_limit}"
        )
    return _semaphore
