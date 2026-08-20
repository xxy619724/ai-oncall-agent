"""异步任务 API 路由

提供完整的异步任务系统接口（阶段一）：
- POST /api/tasks           提交任务（返回 202 + task_id）
- GET  /api/tasks           列出最近任务
- GET  /api/tasks/{task_id} 查询单个任务状态
- GET  /api/tasks/{task_id}/stream  SSE 流式获取任务事件
- POST /api/tasks/{task_id}/cancel  取消任务

与现有 /api/aiops 双轨并存，互不影响。
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.models.task import TaskPriority, TaskStatus, TransitionError
from app.services.task_service import get_task_service
from app.services.task_worker import event_store


router = APIRouter()


# ============================================================
# 请求/响应模型
# ============================================================

class TaskSubmitRequest(BaseModel):
    """提交任务请求"""
    input_text: str = Field(..., description="用户输入任务描述")
    session_id: str | None = Field(
        None, description="会话 ID（可选，不传则用 task_id 代替）"
    )
    priority: str | None = Field(
        "normal",
        description="任务优先级：high（交互式对话）/ normal（默认）/ low（批量任务）",
    )


class TaskCancelResponse(BaseModel):
    """取消任务响应"""
    task_id: str
    status: str
    message: str


# ============================================================
# API 路由
# ============================================================

@router.post("/tasks")
async def submit_task(request: TaskSubmitRequest):
    """提交异步任务

    流程：
    1. 创建任务记录（status=created）
    2. 入队（status=queued）
    3. 立即返回 202 + task_id（不等执行完成）

    客户端拿到 task_id 后可通过以下方式获取结果：
    - GET /api/tasks/{task_id}          轮询查询状态
    - GET /api/tasks/{task_id}/stream   SSE 流式获取事件

    Returns:
        202 Accepted + {task_id, status}
    """
    try:
        service = get_task_service()

        # 队列已满检查
        if service.queue.full():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "任务队列已满，请稍后重试",
                },
            )

        task = await service.submit(
            input_text=request.input_text,
            session_id=request.session_id,
            priority=TaskPriority.from_str(request.priority),
        )

        return JSONResponse(
            status_code=202,
            content={
                "task_id": task.task_id,
                "status": task.status.value,
                "priority": task.priority.name.lower(),
                "message": "任务已接收，正在后台执行",
            },
        )

    except Exception as e:
        logger.error(f"提交任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_tasks(limit: int = 20):
    """列出最近任务

    Args:
        limit: 返回数量上限（默认 20）

    Returns:
        任务列表
    """
    try:
        service = get_task_service()
        tasks = await service.list_tasks(limit=limit)
        return {
            "status": "success",
            "count": len(tasks),
            "data": [t.to_dict() for t in tasks],
        }
    except Exception as e:
        logger.error(f"列出任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """查询单个任务状态

    Args:
        task_id: 任务 ID

    Returns:
        任务详情（含状态、进度、结果）
    """
    try:
        service = get_task_service()
        task = await service.get(task_id)
        if task is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": f"任务 {task_id} 不存在",
                },
            )
        return {"status": "success", "data": task.to_dict()}
    except Exception as e:
        logger.error(f"查询任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/stream")
async def stream_task_events(task_id: str):
    """SSE 流式获取任务事件

    从内存 event_store 读取任务执行事件，推送给客户端。
    任务结束后自动关闭流。

    适用场景：
    - 提交任务后实时观察执行进度
    - 任务可能已完成（直接推送历史事件 + complete）
    - 任务可能正在执行（推送已有事件 + 等待新事件）

    Args:
        task_id: 任务 ID

    Returns:
        SSE 事件流
    """
    try:
        service = get_task_service()

        # 校验任务存在
        task = await service.get(task_id)
        if task is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": f"任务 {task_id} 不存在",
                },
            )

        async def event_generator():
            """SSE 事件生成器"""
            try:
                # 推送任务初始状态
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "task_status",
                        "stage": "task_status",
                        "status": task.status.value,
                        "message": f"任务当前状态: {task.status.value}",
                    }, ensure_ascii=False),
                }

                # 如果任务已结束（终态），直接推送结果并关闭
                if TaskStatus.is_terminal(task.status):
                    final_event = {
                        "type": (
                            "complete" if task.status == TaskStatus.SUCCEEDED
                            else task.status.value
                        ),
                        "stage": "final",
                        "status": task.status.value,
                        "message": "任务已结束",
                        "result": task.result_text,
                        "error": task.error_message,
                    }
                    yield {
                        "event": "message",
                        "data": json.dumps(final_event, ensure_ascii=False),
                    }
                    return

                # 任务未结束：从 event_store 流式读取事件
                notifier = event_store.get_notifier(task_id)
                consumed_index = 0

                while True:
                    # 必须先 clear 再读：若放到读之后，Worker 在
                    # 「读完」与「clear」之间 append 的那次通知会被抹掉，
                    # 导致本轮 wait() 空等满超时才恢复。
                    # 先 clear 的话，这期间的 append 会重新 set，
                    # 最坏情况只是多空转一轮，不会丢通知。
                    notifier.clear()

                    # 读取新事件
                    new_events = event_store.get_events(
                        task_id, after_index=consumed_index
                    )
                    for evt in new_events:
                        yield {
                            "event": "message",
                            "data": json.dumps(evt, ensure_ascii=False),
                        }
                    consumed_index += len(new_events)

                    # 检查是否结束
                    if event_store.is_complete(task_id):
                        break

                    # 兜底：任务已进终态但终止事件缺失（如事件缓冲被清理），
                    # 直接补一条 final 事件收尾，避免这里无限发心跳不退出
                    latest = await service.get(task_id)
                    if latest is not None and TaskStatus.is_terminal(latest.status):
                        yield {
                            "event": "message",
                            "data": json.dumps({
                                "type": (
                                    "complete"
                                    if latest.status == TaskStatus.SUCCEEDED
                                    else latest.status.value
                                ),
                                "stage": "final",
                                "status": latest.status.value,
                                "message": "任务已结束",
                                "result": latest.result_text,
                                "error": latest.error_message,
                            }, ensure_ascii=False),
                        }
                        break

                    # 等待新事件（带超时，避免长时间阻塞）
                    try:
                        await asyncio.wait_for(notifier.wait(), timeout=30.0)
                    except asyncio.TimeoutError:
                        # 推送心跳保活
                        yield {
                            "event": "message",
                            "data": json.dumps({
                                "type": "heartbeat",
                                "message": "keep-alive",
                            }, ensure_ascii=False),
                        }

                logger.info(f"[Task {task_id[:8]}] SSE 流式响应完成")

            except Exception as e:
                logger.error(
                    f"[Task {task_id[:8]}] SSE 流式响应异常: {e}",
                    exc_info=True,
                )
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "error",
                        "stage": "stream_error",
                        "message": f"流式响应异常: {str(e)}",
                    }, ensure_ascii=False),
                }

        return EventSourceResponse(event_generator())

    except Exception as e:
        logger.error(f"启动任务流式响应失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务

    - QUEUED 状态：直接取消
    - RUNNING 状态：发送取消信号，Worker 检测后退出
    - 终态：返回 409 Conflict

    Args:
        task_id: 任务 ID

    Returns:
        取消后的任务状态
    """
    try:
        service = get_task_service()
        task = await service.cancel(task_id)

        if task is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": f"任务 {task_id} 不存在",
                },
            )

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "message": (
                "任务已取消"
                if task.status == TaskStatus.CANCELLED
                else "取消信号已发送，Worker 将在下一个检查点退出"
            ),
        }

    except TransitionError as e:
        # 终态任务不可取消
        return JSONResponse(
            status_code=409,
            content={
                "task_id": task_id,
                "status": e.from_status.value,
                "message": f"任务已处于终态（{e.from_status.value}），无法取消",
            },
        )
    except Exception as e:
        logger.error(f"取消任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
