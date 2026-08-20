"""经验记忆 TTL 定时软删除服务

参考 checkpoint_cleaner.py 的设计模式，扫描 aiops_experiences 表中
过期的经验记录（created_at + ttl_days < now），标记为 deprecated（软删除）。

设计要点:
1. 软删除不物理删除：保留审计追溯，Milvus 向量保留靠 expr 过滤不召回
2. 每条经验有独立 TTL：通过 ttl_days 字段动态判断，不是全局统一 cutoff
3. 分批处理：避免长时间锁库，与 checkpoint_cleaner 一致
4. 降级安全：定时任务异常只记日志，不影响主流程

闭环关系:
- TTL 定时任务 → 标记过期经验 status=deprecated
- expr 过滤（vector_search_service）→ 检索时跳过 deprecated
- 两者配合，TTL 软删除才真正闭环
"""

import asyncio
from datetime import datetime
from typing import Optional

import aiosqlite
from loguru import logger

from app.config import config


class ExperienceTtlCleaner:
    """经验记忆 TTL 定时软删除器

    策略:
    1. 定时扫描 aiops_experiences 表中 status='active' 的记录
    2. 判断 created_at + ttl_days < now（每条经验独立 TTL）
    3. 过期记录标记 status='deprecated'（不物理删除）
    4. Milvus 向量保留，靠 expr 过滤不再召回
    5. 分批处理，避免长时间锁库
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        default_ttl_days: Optional[int] = None,
        cleanup_interval_hours: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.conn = conn
        self.default_ttl_days = (
            default_ttl_days or config.memory_default_ttl_days
        )
        self.cleanup_interval = (
            cleanup_interval_hours or config.memory_ttl_cleanup_interval_hours
        ) * 3600
        self.batch_size = (
            batch_size or config.memory_ttl_cleanup_batch_size
        )
        self._task: Optional[asyncio.Task] = None

    async def cleanup_expired(self) -> int:
        """扫描过期经验标记 deprecated（软删除）

        过期判断：julianday(now) - julianday(created_at) > ttl_days
        即创建时间 + TTL 天数 早于当前时间，视为过期。

        Returns:
            标记为 deprecated 的经验数量
        """
        now = datetime.now()
        now_str = now.isoformat()

        logger.info(
            f"开始扫描过期经验: now={now_str}, "
            f"default_ttl={self.default_ttl_days}天, batch_size={self.batch_size}"
        )

        # 查找 status='active' 且过期的经验
        # SQLite 用 julianday 函数算日期差，每条经验独立 ttl_days
        cursor = await self.conn.execute(
            """
            SELECT id, created_at, ttl_days FROM aiops_experiences
            WHERE status = 'active'
              AND ttl_days IS NOT NULL
              AND created_at IS NOT NULL
              AND julianday(?) - julianday(created_at) > ttl_days
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (now_str, self.batch_size),
        )
        expired_rows = [row async for row in cursor]

        if not expired_rows:
            logger.info("无过期经验需标记 deprecated")
            return 0

        marked = 0
        for row in expired_rows:
            exp_id, created_at, ttl_days = row[0], row[1], row[2]
            try:
                await self.conn.execute(
                    "UPDATE aiops_experiences SET status = 'deprecated' WHERE id = ?",
                    (exp_id,),
                )
                marked += 1
                logger.debug(
                    f"经验标记 deprecated: id={exp_id}, "
                    f"created_at={created_at}, ttl_days={ttl_days}"
                )
            except Exception as e:
                logger.error(
                    f"标记过期经验 deprecated 失败 id={exp_id}: {e}"
                )

        await self.conn.commit()
        logger.info(
            f"TTL 过期标记完成: {marked}/{len(expired_rows)} 条经验标记为 deprecated"
        )
        return marked

    async def get_stats(self) -> dict:
        """获取 TTL 清理统计信息（用于监控）

        Returns:
            统计字典：
            - active_count: 当前 active 经验数
            - expired_pending: 已过期但还没标记 deprecated 的数量
            - deprecated_count: 已标记 deprecated 的经验数
            - default_ttl_days: 默认 TTL 配置
        """
        now_str = datetime.now().isoformat()

        # active 经验总数
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM aiops_experiences WHERE status = 'active'"
        )
        row = await cursor.fetchone()
        active_count = row[0] if row else 0

        # 已过期待标记数量
        cursor = await self.conn.execute(
            """
            SELECT COUNT(*) FROM aiops_experiences
            WHERE status = 'active'
              AND ttl_days IS NOT NULL
              AND created_at IS NOT NULL
              AND julianday(?) - julianday(created_at) > ttl_days
            """,
            (now_str,),
        )
        row = await cursor.fetchone()
        expired_pending = row[0] if row else 0

        # deprecated 经验数
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM aiops_experiences WHERE status = 'deprecated'"
        )
        row = await cursor.fetchone()
        deprecated_count = row[0] if row else 0

        return {
            "active_count": active_count,
            "expired_pending_deprecate": expired_pending,
            "deprecated_count": deprecated_count,
            "default_ttl_days": self.default_ttl_days,
        }

    async def start_periodic_cleanup(self) -> None:
        """启动定时清理后台任务（FastAPI lifespan 调用）

        模式与 SqliteCheckpointCleaner.start_periodic_cleanup 一致：
        - 启动后立即跑一次（验证流程）
        - 之后按 cleanup_interval 间隔循环
        - 异常只记日志，不退出循环
        """
        # 启动时立即跑一次，验证流程可用
        try:
            await self.cleanup_expired()
        except Exception as e:
            logger.error(f"首次 TTL 清理异常: {e}", exc_info=True)

        async def _loop():
            logger.info(
                f"经验 TTL 定时清理任务已启动: 间隔={self.cleanup_interval}s, "
                f"默认TTL={self.default_ttl_days}天, batch={self.batch_size}"
            )
            while True:
                await asyncio.sleep(self.cleanup_interval)
                try:
                    await self.cleanup_expired()
                except Exception as e:
                    logger.error(
                        f"经验 TTL 定时清理异常: {e}", exc_info=True
                    )

        self._task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        """停止定时清理任务（FastAPI lifespan shutdown 调用）"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("经验 TTL 定时清理任务已停止")
