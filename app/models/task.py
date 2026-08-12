"""异步任务数据模型 + 状态机定义

阶段一：最小可用异步任务系统（6 种状态）

状态机：
                    ┌─────────── cancelled
                    │
    created → queued → running → succeeded
                    │
                    └──→ failed

合法流转：
- created → queued         （入队）
- queued  → running        （Worker 拉取）
- queued  → cancelled      （用户取消排队中任务）
- running → succeeded      （执行成功）
- running → failed         （执行异常）
- running → cancelled      （用户取消执行中任务）

终态：succeeded / failed / cancelled（不可再流转）
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    """任务状态枚举（6 种状态）"""

    CREATED = "created"        # 已创建，未入队
    QUEUED = "queued"          # 已入队，等待 Worker 消费
    RUNNING = "running"        # Worker 正在执行
    SUCCEEDED = "succeeded"    # 执行成功（终态）
    FAILED = "failed"          # 执行失败（终态）
    CANCELLED = "cancelled"    # 用户取消（终态）

    @classmethod
    def is_terminal(cls, status: "TaskStatus") -> bool:
        """判断是否为终态（不可再流转）"""
        return status in (cls.SUCCEEDED, cls.FAILED, cls.CANCELLED)


class TaskPriority(Enum):
    """任务优先级（数字越小越优先）

    用于 PriorityQueue 排序：
    - HIGH（0）：用户交互式对话，需快速响应
    - NORMAL（1）：默认优先级
    - LOW（2）：批量任务、测试任务、后台巡检
    """

    HIGH = 0
    NORMAL = 1
    LOW = 2

    @classmethod
    def from_str(cls, value: str | None) -> "TaskPriority":
        """从字符串解析优先级（不区分大小写，默认 NORMAL）"""
        if value is None:
            return cls.NORMAL
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.NORMAL


# 状态流转合法性表：key 允许流转到 value 中的任一状态
# 终态状态不在此表中（即不允许流转）
_VALID_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.QUEUED}),
    TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
}


def is_valid_transition(from_status: TaskStatus, to_status: TaskStatus) -> bool:
    """校验状态流转是否合法

    Args:
        from_status: 当前状态
        to_status: 目标状态

    Returns:
        是否允许流转
    """
    allowed = _VALID_TRANSITIONS.get(from_status, frozenset())
    return to_status in allowed


class TransitionError(Exception):
    """非法状态流转异常"""

    def __init__(self, from_status: TaskStatus, to_status: TaskStatus):
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"非法状态流转: {from_status.value} → {to_status.value}"
        )


@dataclass
class Task:
    """任务数据模型（对应 SQLite tasks 表）"""

    task_id: str                           # UUID4，全局唯一
    session_id: str                        # 关联会话（兼容现有 session_id 体系）
    input_text: str                        # 用户输入任务描述
    status: TaskStatus = TaskStatus.CREATED
    priority: TaskPriority = TaskPriority.NORMAL  # 任务优先级
    progress_completed: int = 0            # 已完成步骤数
    progress_total: int = 0                # 总步骤数（plan 阶段后更新）
    result_text: Optional[str] = None      # 最终响应文本（阶段一存全文，阶段二改 result_ref）
    error_message: Optional[str] = None    # 失败原因
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None       # Worker 开始执行时间
    ended_at: Optional[str] = None         # 执行结束时间（成功/失败/取消）

    def to_dict(self) -> dict:
        """转换为 dict（API 响应用）"""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "input_text": self.input_text,
            "status": self.status.value,
            "priority": self.priority.name.lower(),  # high / normal / low
            "progress": {
                "completed": self.progress_completed,
                "total": self.progress_total,
            },
            "result_text": self.result_text,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "is_terminal": TaskStatus.is_terminal(self.status),
        }
