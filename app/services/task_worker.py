"""TaskWorker - 后台任务执行 Worker

职责：
1. 从 task_service.queue 消费 task_id
2. 状态流转：QUEUED → RUNNING
3. 调用 aiops_service.execute() 流式执行
4. 收集事件到内存 event_store（供 SSE 端点读取）
5. 检测取消信号，优雅退出
6. 执行完成：RUNNING → SUCCEEDED / FAILED / CANCELLED

设计要点：
- 单 Worker（阶段一），asyncio.create_task 启动
- 任务级硬超时（config.task_timeout_seconds，默认 300s）
- 取消信号：asyncio.Event，每个事件循环迭代检查
- 事件存储：内存 dict[task_id, list[event]]，阶段一不持久化
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from app.config import config
from app.models.task import TaskStatus
from app.services.task_service import TaskService, get_task_service


# 终止事件类型：出现其一即代表任务已结束，SSE 可以关闭
TERMINAL_EVENT_TYPES = ("complete", "error", "cancelled")


class EventStore:
    """任务事件内存存储（阶段一：不持久化）

    每个 task_id 对应一个事件列表，供 SSE 端点流式读取。
    阶段二会替换为独立 EventStore 表。
    """

    def __init__(self, max_size_per_task: int = 200):
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._max_size = max_size_per_task
        # 新事件通知：task_id → asyncio.Event，SSE 端点等待
        self._notifiers: dict[str, asyncio.Event] = {}
        # 已丢弃事件计数（缓冲区满时），用于给客户端一个明确提示
        self._dropped: dict[str, int] = {}

    def append(self, task_id: str, event: dict[str, Any]) -> None:
        """追加事件

        缓冲区满时丢弃中间事件，但终止事件（complete/error/cancelled）必须留下：
        SSE 端点靠它判断流是否结束，一旦丢掉就会无限发心跳直到客户端断开。
        """
        if task_id not in self._events:
            self._events[task_id] = []
        events = self._events[task_id]

        is_terminal = event.get("type") in TERMINAL_EVENT_TYPES

        if len(events) < self._max_size:
            events.append(event)
        elif is_terminal:
            # 腾一个位置给终止事件：挤掉最老的中间事件
            # （首个事件通常是状态说明，丢中间的对客户端影响最小）
            drop_at = len(events) // 2
            events.pop(drop_at)
            self._dropped[task_id] = self._dropped.get(task_id, 0) + 1
            events.append(event)
        else:
            self._dropped[task_id] = self._dropped.get(task_id, 0) + 1

        # 通知等待的 SSE 消费者
        notifier = self._notifiers.get(task_id)
        if notifier is not None:
            notifier.set()

    def dropped_count(self, task_id: str) -> int:
        """返回因缓冲区满而丢弃的事件数（0 表示无丢弃）"""
        return self._dropped.get(task_id, 0)

    def get_events(
        self, task_id: str, after_index: int = 0
    ) -> list[dict[str, Any]]:
        """获取指定 task 的事件（从 after_index 开始）"""
        events = self._events.get(task_id, [])
        return events[after_index:]

    def get_notifier(self, task_id: str) -> asyncio.Event:
        """获取事件通知器（SSE 端点等待新事件）"""
        if task_id not in self._notifiers:
            self._notifiers[task_id] = asyncio.Event()
        return self._notifiers[task_id]

    def is_complete(self, task_id: str) -> bool:
        """任务是否已结束（最后一个事件是 complete/error/cancelled）"""
        events = self._events.get(task_id, [])
        if not events:
            return False
        last = events[-1]
        return last.get("type") in TERMINAL_EVENT_TYPES

    def cleanup(self, task_id: str) -> None:
        """清理任务事件（任务结束后延迟调用）"""
        self._events.pop(task_id, None)
        self._notifiers.pop(task_id, None)
        self._dropped.pop(task_id, None)

    def task_count(self) -> int:
        """当前驻留内存的任务数（用于观察是否泄漏）"""
        return len(self._events)


# 全局 EventStore 单例
event_store = EventStore(max_size_per_task=config.task_event_buffer_size)


class TaskWorker:
    """后台任务执行 Worker

    用法：
        worker = TaskWorker(task_service)
        await worker.start()  # 在 FastAPI lifespan 中启动
        ...
        await worker.stop()   # 在 lifespan shutdown 中停止
    """

    def __init__(self, task_service: TaskService):
        self.task_service = task_service
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # 延迟清理协程的强引用集合：asyncio 只持弱引用，不留强引用的话
        # 任务可能在执行前被 GC 回收，清理就静默不发生了
        self._cleanup_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        """启动 Worker 后台协程"""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"TaskWorker 已启动 (concurrency={config.task_worker_concurrency}, "
            f"timeout={config.task_timeout_seconds}s)"
        )

    async def stop(self) -> None:
        """停止 Worker（优雅退出）"""
        if self._task is None:
            return
        self._running = False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

        # 取消仍在等待的延迟清理协程（其 CancelledError 分支会立即完成清理）
        pending = list(self._cleanup_tasks)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._cleanup_tasks.clear()

        logger.info("TaskWorker 已停止")

    async def _run_loop(self) -> None:
        """Worker 主循环：从优先级队列消费任务并执行"""
        logger.info("TaskWorker 主循环已启动，等待任务...")
        while self._running:
            try:
                # 从 PriorityQueue 拉取：(priority_value, sequence, task_id)
                priority_value, sequence, task_id = (
                    await self.task_service.queue.get()
                )
                logger.info(
                    f"[Task {task_id[:8]}] Worker 拉取任务 "
                    f"(priority={priority_value}, seq={sequence})"
                )
            except asyncio.CancelledError:
                break

            try:
                await self._execute_task(task_id)
            except Exception as e:
                logger.error(
                    f"[Task {task_id[:8]}] Worker 执行异常: {e}",
                    exc_info=True,
                )
                # 兜底：确保标记为 failed
                try:
                    await self.task_service.transition_to_failed(
                        task_id, f"Worker 异常: {e}"
                    )
                except Exception as inner:
                    # 状态已是终态等情形下流转会失败，此处仅记录，不再往上抛
                    logger.warning(
                        f"[Task {task_id[:8]}] 兜底标记 failed 未成功: {inner}"
                    )
            finally:
                self.task_service.queue.task_done()
                # 无论成功/失败/取消，都要释放该任务占用的内存
                self._schedule_cleanup(task_id)

    def _schedule_cleanup(self, task_id: str) -> None:
        """延迟清理任务的事件缓冲与取消信号

        延迟而非立即清理：SSE 端点可能仍在读取事件，立即清空会让消费者
        拿不到终止事件。保留 task_event_retention_seconds 后释放。
        """

        async def _delayed() -> None:
            try:
                await asyncio.sleep(config.task_event_retention_seconds)
                dropped = event_store.dropped_count(task_id)
                if dropped:
                    logger.warning(
                        f"[Task {task_id[:8]}] 事件缓冲区溢出，丢弃 {dropped} 条中间事件"
                        f"（上限 {config.task_event_buffer_size}）"
                    )
                event_store.cleanup(task_id)
                self.task_service.cleanup_cancel_event(task_id)
                logger.debug(
                    f"[Task {task_id[:8]}] 事件缓冲已释放"
                    f"（内存驻留任务数={event_store.task_count()}）"
                )
            except asyncio.CancelledError:
                # 关服时取消：立即清理，不再等待
                event_store.cleanup(task_id)
                self.task_service.cleanup_cancel_event(task_id)
                raise

        t = asyncio.create_task(_delayed())
        # 持强引用，并在完成后移除，避免集合本身变成泄漏点
        self._cleanup_tasks.add(t)
        t.add_done_callback(self._cleanup_tasks.discard)

    async def _execute_task(self, task_id: str) -> None:
        """执行单个任务

        流程：
        1. 注册取消信号
        2. 状态流转 QUEUED → RUNNING
        3. 调用 aiops_service.execute() 流式执行（带超时）
        4. 收集事件到 event_store
        5. 检测取消信号 → CANCELLED
        6. 执行完成 → SUCCEEDED / FAILED
        """
        # 排队期间被取消的任务仍留在队列里，会被 Worker 正常拉到。
        # 此时状态已是 CANCELLED（终态），直接跳过，否则下面的
        # transition_to_running 会抛 TransitionError 刷出误导性错误日志。
        existing = await self.task_service.get(task_id)
        if existing is None:
            logger.error(f"[Task {task_id[:8]}] 任务不存在，跳过")
            return
        if TaskStatus.is_terminal(existing.status):
            logger.info(
                f"[Task {task_id[:8]}] 拉取到已处于终态的任务"
                f"（{existing.status.value}），跳过执行"
            )
            # 补一条终止事件，让已连上的 SSE 消费者能正常收尾
            if not event_store.is_complete(task_id):
                event_store.append(task_id, {
                    "type": existing.status.value,
                    "stage": "skipped",
                    "message": f"任务在排队期间已{existing.status.value}",
                })
            return

        # 注册取消信号
        cancel_event = self.task_service.register_cancel_event(task_id)

        # 状态流转：QUEUED → RUNNING
        task = await self.task_service.transition_to_running(task_id)
        if task is None:
            logger.error(f"[Task {task_id[:8]}] 任务不存在，跳过")
            return

        # 延迟导入避免循环依赖
        from app.services.aiops_service import aiops_service

        # 执行任务（带超时 + 取消检测）
        result_text = ""
        progress_completed = 0
        progress_total = 0
        was_cancelled = False
        error_message = ""

        try:
            # 用 asyncio.wait_for 实现硬超时
            async def _run_with_cancel_check():
                nonlocal result_text, progress_completed, progress_total, was_cancelled

                async for event in aiops_service.execute(
                    user_input=task.input_text,
                    session_id=task.session_id,
                ):
                    # 检测取消信号
                    if cancel_event.is_set():
                        was_cancelled = True
                        # 推送取消事件到 event_store
                        event_store.append(task_id, {
                            "type": "cancelled",
                            "stage": "cancelled",
                            "message": "任务已被用户取消",
                        })
                        break

                    # 收集事件到 event_store
                    event_store.append(task_id, event)

                    # 提取进度信息
                    if event.get("type") == "plan":
                        plan = event.get("plan", [])
                        progress_total = len(plan)
                    elif event.get("type") == "step_complete":
                        progress_completed += 1
                    elif event.get("type") == "complete":
                        result_text = event.get("response", "")

                # 检查任务是否完成
                if not was_cancelled and not result_text:
                    # 没拿到 complete 事件，尝试获取最终状态
                    pass

            await asyncio.wait_for(
                _run_with_cancel_check(),
                timeout=config.task_timeout_seconds,
            )

        except asyncio.TimeoutError:
            error_message = (
                f"任务执行超时（>{config.task_timeout_seconds}s）"
            )
            logger.error(f"[Task {task_id[:8]}] {error_message}")
            await self.task_service.transition_to_failed(
                task_id, error_message
            )
            event_store.append(task_id, {
                "type": "error",
                "stage": "timeout",
                "message": error_message,
            })
            return

        except Exception as e:
            error_message = f"任务执行失败: {e}"
            logger.error(
                f"[Task {task_id[:8]}] {error_message}", exc_info=True
            )
            await self.task_service.transition_to_failed(
                task_id, error_message
            )
            event_store.append(task_id, {
                "type": "error",
                "stage": "error",
                "message": error_message,
            })
            return

        # 处理取消
        if was_cancelled:
            await self.task_service.transition_to_cancelled(task_id)
            logger.info(f"[Task {task_id[:8]}] 任务已取消")
            return

        # 执行成功
        if progress_total == 0 and progress_completed > 0:
            progress_total = progress_completed  # 兜底

        await self.task_service.transition_to_succeeded(
            task_id,
            result_text=result_text,
            progress_completed=progress_completed,
            progress_total=progress_total,
        )
        logger.info(
            f"[Task {task_id[:8]}] 任务执行成功 "
            f"(steps={progress_completed}/{progress_total})"
        )


# ============================================================
# 全局 Worker 单例
# ============================================================

_task_worker: Optional[TaskWorker] = None


async def start_task_worker() -> TaskWorker:
    """启动全局 TaskWorker（在 main.py lifespan 中调用）"""
    global _task_worker
    if _task_worker is not None:
        return _task_worker

    service = get_task_service()
    _task_worker = TaskWorker(service)
    await _task_worker.start()
    return _task_worker


async def stop_task_worker() -> None:
    """停止全局 TaskWorker（在 main.py lifespan shutdown 中调用）"""
    global _task_worker
    if _task_worker is not None:
        await _task_worker.stop()
        _task_worker = None
