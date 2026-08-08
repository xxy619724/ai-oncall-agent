"""双层记忆服务 - SQLite 持久真相 + Redis 摘要快照

提供统一的记忆管理接口：
- 读取: Redis 摘要（快速上下文）+ SQLite 近期消息（完整历史）
- 写入: LangGraph checkpointer 自动写入 SQLite，摘要通过 sync_summary 同步到 Redis
- 清空: 双层清空
- 降级: Redis 不可用时自动跳过，纯 SQLite 可用
"""

from typing import Any, Optional

import redis
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from loguru import logger

from app.config import config


class MemoryService:
    """双层记忆服务：SQLite 持久真相 + Redis 摘要快照"""

    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        """初始化记忆服务

        Args:
            checkpointer: LangGraph checkpointer 实例（SqliteSaver 或 MemorySaver）
        """
        self.checkpointer = checkpointer
        self._redis: Optional[redis.Redis] = None
        self._redis_available = False

    def init_redis(self) -> None:
        """初始化 Redis 连接，失败时标记不可用（降级模式）"""
        try:
            self._redis = redis.from_url(
                config.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # 测试连接
            self._redis.ping()
            self._redis_available = True
            logger.info(f"Redis 连接成功: {config.redis_url}")
        except Exception as e:
            self._redis = None
            self._redis_available = False
            logger.warning(f"Redis 连接失败，降级为纯 SQLite 模式: {e}")

    def close_redis(self) -> None:
        """关闭 Redis 连接"""
        if self._redis is not None:
            try:
                self._redis.close()
                logger.info("Redis 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭 Redis 连接失败: {e}")

    def _get_summary(self, session_id: str) -> Optional[str]:
        """从 Redis 获取摘要，失败时返回 None（降级）

        Args:
            session_id: 会话 ID

        Returns:
            摘要文本，或 None（无摘要 / Redis 不可用）
        """
        if not self._redis_available or self._redis is None:
            return None

        try:
            key = f"session:{session_id}:summary"
            value = self._redis.get(key)
            if value:
                logger.debug(f"从 Redis 获取摘要: {session_id}, 长度={len(value)}")
                return value
            return None
        except Exception as e:
            logger.warning(f"从 Redis 获取摘要失败: {e}")
            return None

    def _save_summary(self, session_id: str, summary: str) -> None:
        """保存摘要到 Redis，失败时记录日志（降级）

        Args:
            session_id: 会话 ID
            summary: 摘要文本
        """
        if not self._redis_available or self._redis is None:
            return

        try:
            key = f"session:{session_id}:summary"
            self._redis.setex(key, config.redis_summary_ttl, summary)
            logger.debug(f"摘要已保存到 Redis: {session_id}, 长度={len(summary)}")
        except Exception as e:
            logger.warning(f"保存摘要到 Redis 失败: {e}")

    async def _aget_messages(self, session_id: str) -> list[BaseMessage]:
        """从 checkpointer 异步获取会话消息列表

        使用 AsyncSqliteSaver 的 aget_tuple 异步接口。注意：不能在事件循环
        线程中同步调用 AsyncSqliteSaver.get()，否则会死锁（其内部 async 操作
        需要事件循环，而事件循环正被当前同步调用阻塞）。

        Args:
            session_id: 会话 ID（即 thread_id）

        Returns:
            消息列表，无历史时返回空列表
        """
        try:
            config_dict = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = await self.checkpointer.aget_tuple(config_dict)

            if not checkpoint_tuple:
                return []

            checkpoint_data = (
                checkpoint_tuple.checkpoint
                if hasattr(checkpoint_tuple, "checkpoint")
                else {}
            )

            if not isinstance(checkpoint_data, dict):
                return []

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            return messages if isinstance(messages, list) else []
        except Exception as e:
            logger.error(f"从 checkpointer 获取消息失败: {session_id}, 错误: {e}")
            return []

    def _extract_summary(self, messages: list[BaseMessage]) -> Optional[str]:
        """从消息列表中提取 SummarizationMiddleware 生成的摘要

        SummarizationMiddleware 压缩后会在消息列表头部插入一个 SystemMessage，
        内容是旧对话的摘要。通过排除原始系统提示词来识别摘要消息。

        Args:
            messages: 消息列表

        Returns:
            摘要文本，或 None（无摘要）
        """
        system_messages = [
            msg for msg in messages if isinstance(msg, SystemMessage)
        ]

        # 如果只有 1 个或 0 个 SystemMessage，说明没有摘要
        if len(system_messages) <= 1:
            return None

        # 原始系统提示词通常是第一个 SystemMessage
        # 摘要是 SummarizationMiddleware 额外插入的 SystemMessage
        for msg in system_messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            # 原始系统提示词以 "你是一个专业的AI助手" 开头
            if not str(content).startswith("你是一个专业的AI助手"):
                return str(content)

        return None

    async def aget_context(self, session_id: str) -> list[BaseMessage]:
        """获取会话上下文：Redis 摘要 + SQLite 近期消息

        读取流程：
        1. 查 Redis 摘要（快速, ~1ms）
        2. 查 SQLite 完整历史消息
        3. 拼装：[摘要 SystemMessage] + [历史消息]

        Args:
            session_id: 会话 ID

        Returns:
            拼装后的消息列表
        """
        # 1. 查 Redis 摘要
        summary = self._get_summary(session_id)

        # 2. 查 SQLite 消息
        messages = await self._aget_messages(session_id)

        # 3. 拼装
        if summary:
            return [SystemMessage(content=summary)] + messages
        return messages

    async def sync_summary(self, session_id: str) -> None:
        """ainvoke 完成后调用：从 SQLite 提取摘要写入 Redis

        从 checkpointer 读取最新消息列表，检测 SummarizationMiddleware
        生成的摘要 SystemMessage，写入 Redis。

        Args:
            session_id: 会话 ID
        """
        messages = await self._aget_messages(session_id)
        if not messages:
            return

        summary = self._extract_summary(messages)
        if summary:
            self._save_summary(session_id, summary)

    async def aget_history(self, session_id: str) -> list[dict[str, Any]]:
        """获取会话历史（前端展示用）

        从 checkpointer 读取消息，转换为前端格式。

        Args:
            session_id: 会话 ID

        Returns:
            消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        messages = await self._aget_messages(session_id)
        history: list[dict[str, Any]] = []

        from datetime import datetime

        for msg in messages:
            # 跳过系统消息（包括原始提示词和摘要）
            if isinstance(msg, SystemMessage):
                continue

            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)

            timestamp = getattr(msg, "timestamp", None)
            if not timestamp:
                timestamp = datetime.now().isoformat()

            history.append(
                {"role": role, "content": content, "timestamp": timestamp}
            )

        logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
        return history

    async def aclear(self, session_id: str) -> bool:
        """清空双层存储：删 Redis 摘要 + 删 SQLite 线程

        Args:
            session_id: 会话 ID

        Returns:
            是否成功
        """
        success = True

        # 1. 删 Redis 摘要
        if self._redis_available and self._redis is not None:
            try:
                key = f"session:{session_id}:summary"
                self._redis.delete(key)
                logger.debug(f"已删除 Redis 摘要: {session_id}")
            except Exception as e:
                logger.warning(f"删除 Redis 摘要失败: {e}")
                success = False

        # 2. 删 SQLite 线程（异步）
        try:
            await self.checkpointer.adelete_thread(session_id)
            logger.info(f"已清除会话历史: {session_id}")
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            success = False

        return success


# 全局单例（延迟初始化，需要在外部注入 checkpointer）
memory_service: Optional[MemoryService] = None


def init_memory_service(checkpointer: BaseCheckpointSaver) -> MemoryService:
    """初始化全局 MemoryService 单例

    Args:
        checkpointer: LangGraph checkpointer 实例

    Returns:
        MemoryService 实例
    """
    global memory_service
    memory_service = MemoryService(checkpointer)
    memory_service.init_redis()
    return memory_service


def get_memory_service() -> Optional[MemoryService]:
    """获取全局 MemoryService 实例

    Returns:
        MemoryService 实例，未初始化时返回 None
    """
    return memory_service
