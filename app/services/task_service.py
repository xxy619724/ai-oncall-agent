"""TaskService + TaskStore - 异步任务管理服务

职责：
1. TaskStore：SQLite 持久化任务状态（独立 tasks.db）
2. TaskService：任务创建/查询/取消/状态流转（含状态机校验）
3. 任务恢复：服务重启时扫描 running 状态任务标记为 failed

不包含：
- 任务执行逻辑（在 task_worker.py 中）
- API 路由（在 api/task.py 中）
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional

import aiosqlite
from loguru import logger

from app.config import config
from app.models.task import (
    Task,
    TaskPriority,
    TaskStatus,
    TransitionError,
    is_valid_transition,
)


class TaskStore:
    """任务状态 SQLite 持久化层（异步）"""

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or config.task_db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化数据库连接 + 建表（在 main.py lifespan 中调用）"""
        if self._initialized:
            return
        try:
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id             TEXT PRIMARY KEY,
                    session_id          TEXT NOT NULL,
                    input_text          TEXT NOT NULL,
                    status              TEXT NOT NULL,
                    priority            TEXT DEFAULT 'normal',
                    progress_completed  INTEGER DEFAULT 0,
                    progress_total      INTEGER DEFAULT 0,
                    result_text         TEXT,
                    error_message       TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL,
                    started_at          TEXT,
                    ended_at            TEXT
                )
            """)
            # 兼容旧表：若 priority 列不存在则添加（ALTER TABLE 幂等）
            try:
                await self._conn.execute(
                    "ALTER TABLE tasks ADD COLUMN priority TEXT DEFAULT 'normal'"
                )
            except Exception:
                pass  # 列已存在，忽略
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)"
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
            )
            await self._conn.commit()
            self._initialized = True
            logger.info(f"TaskStore 初始化成功: {self._db_path}")
        except Exception as e:
            logger.error(f"TaskStore 初始化失败: {e}")
            self._conn = None
            self._initialized = False

    @property
    def available(self) -> bool:
        return self._conn is not None and self._initialized

    async def save(self, task: Task) -> None:
        """保存任务（UPSERT：不存在则插入，存在则更新）"""
        if not self.available:
            return
        assert self._conn is not None
        try:
            await self._conn.execute(
                """INSERT INTO tasks
                   (task_id, session_id, input_text, status, priority,
                    progress_completed, progress_total,
                    result_text, error_message,
                    created_at, updated_at, started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    priority = excluded.priority,
                    progress_completed = excluded.progress_completed,
                    progress_total = excluded.progress_total,
                    result_text = excluded.result_text,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at""",
                (
                    task.task_id, task.session_id, task.input_text,
                    task.status.value, task.priority.name.lower(),
                    task.progress_completed, task.progress_total,
                    task.result_text, task.error_message,
                    task.created_at, task.updated_at,
                    task.started_at, task.ended_at,
                ),
            )
            await self._conn.commit()
        except Exception as e:
            logger.error(f"TaskStore.save 失败 (task_id={task.task_id}): {e}")

    async def get(self, task_id: str) -> Optional[Task]:
        """查询单个任务"""
        if not self.available:
            return None
        assert self._conn is not None
        try:
            async with self._conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return None
            return self._row_to_task(row)
        except Exception as e:
            logger.error(f"TaskStore.get 失败 (task_id={task_id}): {e}")
            return None

    async def list_tasks(self, limit: int = 20) -> list[Task]:
        """列出最近的任务（按创建时间倒序）"""
        if not self.available:
            return []
        assert self._conn is not None
        try:
            async with self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
            return [self._row_to_task(r) for r in rows]
        except Exception as e:
            logger.error(f"TaskStore.list 失败: {e}")
            return []

    async def list_by_status(self, status: TaskStatus) -> list[Task]:
        """按状态查询任务（用于任务恢复）"""
        if not self.available:
            return []
        assert self._conn is not None
        try:
            async with self._conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at",
                (status.value,),
            ) as cur:
                rows = await cur.fetchall()
            return [self._row_to_task(r) for r in rows]
        except Exception as e:
            logger.error(f"TaskStore.list_by_status 失败: {e}")
            return []

    async def cleanup(self) -> None:
        """关闭连接"""
        if self._conn is not None:
            try:
                await self._conn.close()
                logger.info("TaskStore 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭 TaskStore 连接失败: {e}")
            finally:
                self._conn = None
                self._initialized = False

    @staticmethod
    def _row_to_task(row) -> Task:
        """数据库行转换为 Task 对象"""
        return Task(
            task_id=row[0],
            session_id=row[1],
            input_text=row[2],
            status=TaskStatus(row[3]),
            priority=TaskPriority.from_str(row[4] if len(row) > 4 else "normal"),
            progress_completed=row[5] if len(row) > 5 else (row[4] or 0),
            progress_total=row[6] if len(row) > 6 else (row[5] or 0),
            result_text=row[7] if len(row) > 7 else row[6],
            error_message=row[8] if len(row) > 8 else row[7],
            created_at=row[9] if len(row) > 9 else row[8],
            updated_at=row[10] if len(row) > 10 else row[9],
            started_at=row[11] if len(row) > 11 else row[10],
            ended_at=row[12] if len(row) > 12 else row[11],
        )


class TaskService:
    """任务管理服务（业务层）

    职责：
    1. 创建任务（生成 task_id，写入 Store，入队）
    2. 查询任务状态
    3. 取消任务（带状态机校验）
    4. 状态流转（带状态机校验）
    5. 任务恢复（重启时清理 running 任务）
    """

    def __init__(self, store: TaskStore):
        self.store = store
        # 优先级队列：Worker 从此队列消费 (priority_value, sequence, task_id)
        # PriorityQueue 按元组排序：priority_value 小的优先，sequence 保证同优先级 FIFO
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=config.task_queue_maxsize
        )
        # 自增序号：保证同优先级任务 FIFO
        self._sequence = 0
        # 取消信号：task_id → asyncio.Event
        # Worker 执行时检查此 Event，被 set 则优雅退出
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def submit(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Task:
        """提交任务

        流程：
        1. 生成 task_id（UUID4）
        2. 创建 Task 对象（status=CREATED）
        3. 写入 TaskStore
        4. 入队（status → QUEUED）
        5. 返回 Task（API 层用其 task_id 返回 202）

        Args:
            input_text: 用户输入任务描述
            session_id: 会话 ID（不传则用 task_id 代替）
            priority: 任务优先级（HIGH/NORMAL/LOW，默认 NORMAL）

        Returns:
            创建的 Task 对象
        """
        task_id = str(uuid.uuid4())
        if session_id is None:
            session_id = task_id  # 未指定 session_id 时用 task_id 代替

        task = Task(
            task_id=task_id,
            session_id=session_id,
            input_text=input_text,
            status=TaskStatus.CREATED,
            priority=priority,
        )

        # 写入 Store
        await self.store.save(task)
        logger.info(
            f"[Task {task_id[:8]}] 任务已创建 (priority={priority.name}): "
            f"{input_text[:80]}"
        )

        # 状态流转：CREATED → QUEUED（必须先流转再入队，避免 Worker 拉取时状态还是 CREATED）
        await self._transition(task_id, TaskStatus.QUEUED)

        # 入队（priority_value, sequence, task_id）
        # sequence 保证同优先级 FIFO
        self._sequence += 1
        await self.queue.put((priority.value, self._sequence, task_id))

        return task

    async def get(self, task_id: str) -> Optional[Task]:
        """查询任务状态"""
        return await self.store.get(task_id)

    async def list_tasks(self, limit: int = 20) -> list[Task]:
        """列出最近任务"""
        return await self.store.list_tasks(limit=limit)

    async def cancel(self, task_id: str) -> Optional[Task]:
        """取消任务

        - QUEUED 状态：直接流转到 CANCELLED
        - RUNNING 状态：set 取消信号，Worker 检测后退出并标记 CANCELLED
        - 终态：拒绝（抛 TransitionError）

        Args:
            task_id: 任务 ID

        Returns:
            更新后的 Task，或 None（任务不存在）
        """
        task = await self.store.get(task_id)
        if task is None:
            return None

        # 终态任务不可取消
        if TaskStatus.is_terminal(task.status):
            raise TransitionError(task.status, TaskStatus.CANCELLED)

        if task.status == TaskStatus.QUEUED:
            # 排队中：直接取消
            await self._transition(task_id, TaskStatus.CANCELLED)
            logger.info(f"[Task {task_id[:8]}] 任务已取消（排队中）")

        elif task.status == TaskStatus.RUNNING:
            # 执行中：设置取消信号，Worker 检测后退出
            event = self._cancel_events.get(task_id)
            if event is None:
                event = asyncio.Event()
                self._cancel_events[task_id] = event
            event.set()
            logger.info(f"[Task {task_id[:8]}] 已发送取消信号（执行中）")

        return await self.store.get(task_id)

    async def _transition(
        self,
        task_id: str,
        to_status: TaskStatus,
        *,
        result_text: Optional[str] = None,
        error_message: Optional[str] = None,
        progress_completed: Optional[int] = None,
        progress_total: Optional[int] = None,
    ) -> Optional[Task]:
        """状态流转（内部方法，带状态机校验）

        Args:
            task_id: 任务 ID
            to_status: 目标状态
            result_text: 最终响应（仅 succeeded 时填）
            error_message: 失败原因（仅 failed 时填）
            progress_completed: 已完成步骤数
            progress_total: 总步骤数

        Returns:
            更新后的 Task，或 None（任务不存在）

        Raises:
            TransitionError: 非法状态流转
        """
        task = await self.store.get(task_id)
        if task is None:
            return None

        # 状态机校验
        if not is_valid_transition(task.status, to_status):
            raise TransitionError(task.status, to_status)

        # 更新字段
        task.status = to_status
        task.updated_at = datetime.now().isoformat()

        if result_text is not None:
            task.result_text = result_text
        if error_message is not None:
            task.error_message = error_message
        if progress_completed is not None:
            task.progress_completed = progress_completed
        if progress_total is not None:
            task.progress_total = progress_total

        # 进入 RUNNING 时记录开始时间
        if to_status == TaskStatus.RUNNING:
            task.started_at = datetime.now().isoformat()

        # 进入终态时记录结束时间
        if TaskStatus.is_terminal(to_status):
            task.ended_at = datetime.now().isoformat()

        await self.store.save(task)
        return task

    async def transition_to_running(self, task_id: str) -> Optional[Task]:
        """Worker 拉取任务后调用：QUEUED → RUNNING"""
        return await self._transition(task_id, TaskStatus.RUNNING)

    async def transition_to_succeeded(
        self, task_id: str, result_text: str,
        progress_completed: int, progress_total: int,
    ) -> Optional[Task]:
        """Worker 执行成功后调用：RUNNING → SUCCEEDED"""
        return await self._transition(
            task_id, TaskStatus.SUCCEEDED,
            result_text=result_text,
            progress_completed=progress_completed,
            progress_total=progress_total,
        )

    async def transition_to_failed(
        self, task_id: str, error_message: str,
    ) -> Optional[Task]:
        """Worker 执行失败后调用：RUNNING → FAILED"""
        return await self._transition(
            task_id, TaskStatus.FAILED, error_message=error_message
        )

    async def transition_to_cancelled(self, task_id: str) -> Optional[Task]:
        """Worker 检测到取消信号后调用：RUNNING → CANCELLED"""
        return await self._transition(task_id, TaskStatus.CANCELLED)

    def register_cancel_event(self, task_id: str) -> asyncio.Event:
        """为任务注册取消信号 Event（Worker 执行前调用）"""
        event = asyncio.Event()
        self._cancel_events[task_id] = event
        return event

    def get_cancel_event(self, task_id: str) -> Optional[asyncio.Event]:
        """获取任务的取消信号（Worker 执行中检查）"""
        return self._cancel_events.get(task_id)

    def cleanup_cancel_event(self, task_id: str) -> None:
        """清理取消信号（Worker 执行结束后调用）"""
        self._cancel_events.pop(task_id, None)

    async def recover_interrupted_tasks(self) -> int:
        """任务恢复：服务重启时清理 RUNNING 状态任务

        服务崩溃/重启后，RUNNING 状态的任务实际已停止执行，
        标记为 FAILED（error_message="服务重启中断"）。

        Returns:
            恢复的任务数量
        """
        running_tasks = await self.store.list_by_status(TaskStatus.RUNNING)
        for task in running_tasks:
            try:
                # 直接更新状态（绕过状态机校验，因为这是恢复场景）
                task.status = TaskStatus.FAILED
                task.error_message = "服务重启中断"
                task.ended_at = datetime.now().isoformat()
                task.updated_at = task.ended_at
                await self.store.save(task)
                logger.warning(
                    f"[Task {task.task_id[:8]}] 已标记为失败（服务重启中断）"
                )
            except Exception as e:
                logger.error(
                    f"[Task {task.task_id[:8]}] 恢复失败: {e}"
                )
        return len(running_tasks)


# ============================================================
# 全局单例（在 main.py lifespan 中初始化）
# ============================================================

task_store = TaskStore()
task_service: Optional[TaskService] = None


def get_task_service() -> TaskService:
    """获取全局 TaskService 实例

    Raises:
        RuntimeError: 未初始化
    """
    if task_service is None:
        raise RuntimeError("TaskService 未初始化，请先调用 init_task_service()")
    return task_service


async def init_task_service() -> TaskService:
    """初始化全局 TaskService（在 main.py lifespan 中调用）"""
    global task_service
    await task_store.initialize()
    task_service = TaskService(task_store)

    # 任务恢复：清理上次崩溃遗留的 RUNNING 任务
    recovered = await task_service.recover_interrupted_tasks()
    if recovered > 0:
        logger.info(f"已恢复 {recovered} 个中断任务（标记为 failed）")

    return task_service


async def cleanup_task_service() -> None:
    """清理 TaskService 资源（在 main.py lifespan shutdown 中调用）"""
    await task_store.cleanup()
