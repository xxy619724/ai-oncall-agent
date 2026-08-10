"""SQLite Checkpoint 自动清理服务

清理 LangGraph AsyncSqliteSaver 的过期全量对话，
符合记忆工程文档第 135 行"LTM 不存储全量原始对话，仅保留摘要"。

清理对象:checkpoints 表 + writes 表 + checkpoint_sessions 辅助表
保留对象:Redis 摘要(TTL 自动过期) + Milvus 知识/经验(永久)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import aiosqlite
from loguru import logger

from app.config import config


class SqliteCheckpointCleaner:
    """SQLite Checkpoint 定时清理器

    策略:
    1. 辅助表 checkpoint_sessions 记录每个 thread_id 的最后活跃时间
    2. 定时扫描 last_active_at < cutoff 的会话
    3. 删除 checkpoints + writes + checkpoint_sessions 中的对应记录
    4. 分批清理，避免长时间锁库
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        max_age_days: Optional[int] = None,
        cleanup_interval_hours: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.conn = conn
        self.max_age_days = max_age_days or config.sqlite_checkpoint_max_age_days
        self.cleanup_interval = (
            cleanup_interval_hours or config.sqlite_cleanup_interval_hours
        ) * 3600
        self.batch_size = batch_size or config.sqlite_cleanup_batch_size
        self._task: Optional[asyncio.Task] = None

    async def init_session_table(self) -> None:
        """初始化辅助表:记录会话活跃时间"""
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_sessions (
                thread_id TEXT PRIMARY KEY,
                last_active_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            )
            """
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoint_sessions_last_active "
            "ON checkpoint_sessions(last_active_at)"
        )
        await self.conn.commit()
        logger.info("checkpoint_sessions 辅助表初始化完成")

    async def touch_session(self, thread_id: str, message_count: int = 0) -> None:
        """更新会话活跃时间(在 agent.ainvoke 后调用)

        使用 UPSERT 语法，不存在则插入，存在则更新 last_active_at。
        失败时只记日志，不影响主流程。
        """
        try:
            now = datetime.now().isoformat()
            await self.conn.execute(
                """
                INSERT INTO checkpoint_sessions (thread_id, last_active_at, created_at, message_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    last_active_at = excluded.last_active_at,
                    message_count = excluded.message_count
                """,
                (thread_id, now, now, message_count),
            )
            await self.conn.commit()
        except Exception as e:
            logger.warning(f"touch_session 失败(thread_id={thread_id}): {e}")

    async def cleanup_expired(self) -> int:
        """清理过期的 checkpoint 全量对话

        Returns:
            清理的会话数量
        """
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        cutoff_str = cutoff.isoformat()

        logger.info(
            f"开始清理过期 checkpoint: cutoff={cutoff_str}, "
            f"max_age={self.max_age_days}天, batch_size={self.batch_size}"
        )

        # 1. 查找过期会话(分批，避免一次性加载太多)
        cursor = await self.conn.execute(
            "SELECT thread_id FROM checkpoint_sessions "
            "WHERE last_active_at < ? "
            "ORDER BY last_active_at ASC LIMIT ?",
            (cutoff_str, self.batch_size),
        )
        expired_threads = [row[0] async for row in cursor]

        if not expired_threads:
            logger.info("无过期 checkpoint 需清理")
            return 0

        cleaned = 0
        for thread_id in expired_threads:
            try:
                # 2. 按顺序删除:writes → checkpoints → checkpoint_sessions
                # writes 表有外键关联 checkpoints，先删 writes
                await self.conn.execute(
                    "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
                )
                await self.conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
                )
                await self.conn.execute(
                    "DELETE FROM checkpoint_sessions WHERE thread_id = ?",
                    (thread_id,),
                )
                cleaned += 1
            except Exception as e:
                logger.error(f"清理 thread_id={thread_id} 失败: {e}")

        await self.conn.commit()
        logger.info(
            f"清理完成: 共清理 {cleaned}/{len(expired_threads)} 个过期会话"
        )
        return cleaned

    async def get_stats(self) -> dict:
        """获取清理统计信息(用于监控)"""
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM checkpoint_sessions"
        )
        row = await cursor.fetchone()
        total_sessions = row[0] if row else 0

        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM checkpoint_sessions WHERE last_active_at < ?",
            (cutoff.isoformat(),),
        )
        row = await cursor.fetchone()
        expired_sessions = row[0] if row else 0

        # checkpoints 表可能不存在(降级 MemorySaver 时)
        try:
            cursor = await self.conn.execute("SELECT COUNT(*) FROM checkpoints")
            row = await cursor.fetchone()
            total_checkpoints = row[0] if row else 0
        except Exception:
            total_checkpoints = -1

        return {
            "total_sessions": total_sessions,
            "expired_sessions": expired_sessions,
            "total_checkpoints": total_checkpoints,
            "max_age_days": self.max_age_days,
        }

    async def start_periodic_cleanup(self) -> None:
        """启动定时清理后台任务(FastAPI lifespan 调用)"""
        await self.init_session_table()

        async def _loop():
            logger.info(
                f"Checkpoint 定时清理任务已启动: 间隔={self.cleanup_interval}s, "
                f"保留={self.max_age_days}天"
            )
            while True:
                await asyncio.sleep(self.cleanup_interval)
                try:
                    await self.cleanup_expired()
                except Exception as e:
                    logger.error(f"定时清理异常: {e}", exc_info=True)

        self._task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        """停止定时清理任务(FastAPI lifespan shutdown 调用)"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Checkpoint 定时清理任务已停止")
