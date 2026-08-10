"""经验回写节点：任务完成后将经验结构化写入 LTM（Milvus + SQLite）

对应记忆工程文档信息流的最后一环：判断是否需要写入 Memory。
- Milvus：经验摘要向量化，用于下次相似任务语义检索
- SQLite：结构化真相存储，用于回查完整经验

设计原则：
- 纯副作用节点，返回 {} 不修改 State
- 失败容错，不阻塞主流程（记录日志后返回）
- 通过工厂函数注入 SQLite 连接，避免节点函数签名受限
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import aiosqlite
from loguru import logger

from .state import PlanExecuteState


# 错误识别关键词（用于从 past_steps 中提取失败步骤）
_ERROR_KEYWORDS = ("失败", "错误", "异常", "error", "failed", "exception", "traceback")


def _extract_errors(past_steps: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """从执行历史中提取失败步骤

    Args:
        past_steps: [(step, result), ...] 执行历史

    Returns:
        失败步骤列表 [{"step": ..., "error": ...}]
    """
    errors: List[Dict[str, str]] = []
    for step, result in past_steps:
        result_str = result if isinstance(result, str) else str(result)
        result_lower = result_str.lower()
        if any(kw in result_lower for kw in _ERROR_KEYWORDS):
            errors.append({"step": step, "error": result_str[:500]})
    return errors


def _build_experience_text(
    input_text: str,
    response: str,
    errors: List[Dict[str, str]],
    past_steps: List[Tuple[str, str]],
) -> str:
    """构建经验摘要文本（用于向量化检索）

    策略：任务 + 最终方案 + 踩坑记录 + 关键步骤结论
    遵循文档"证据化压缩"：记录核心结论，丢弃冗长原始数据。
    """
    parts = [f"## 任务\n{input_text}"]

    if response:
        parts.append(f"## 最终方案\n{response}")

    if past_steps:
        steps_text = "\n".join(
            f"- {step}：{result[:200]}{'...' if len(result) > 200 else ''}"
            for step, result in past_steps
        )
        parts.append(f"## 执行步骤摘要\n{steps_text}")

    if errors:
        error_text = "\n".join(
            f"- 步骤【{e['step']}】失败：{e['error']}" for e in errors
        )
        parts.append(f"## 踩坑记录\n{error_text}")

    return "\n\n".join(parts)


async def _write_to_milvus(
    experience_id: str,
    experience_text: str,
    input_text: str,
    has_error: bool,
    timestamp: str,
) -> bool:
    """写入 Milvus（向量化，用于跨会话语义检索）

    Returns:
        是否写入成功
    """
    try:
        from langchain_core.documents import Document

        from app.services.vector_store_manager import vector_store_manager

        doc = Document(
            page_content=experience_text,
            metadata={
                "_source": "aiops_experience",
                "_file_name": f"aiops_experience_{experience_id}",
                "task_type": "aiops_diagnose",
                "has_error": has_error,
                "created_at": timestamp,
                "experience_id": experience_id,
                "task_preview": input_text[:100],
            },
        )
        vector_store_manager.add_documents([doc])
        logger.info(f"经验已写入 Milvus: id={experience_id}")
        return True
    except Exception as e:
        logger.error(f"写入 Milvus 失败: {e}")
        return False


async def _write_to_sqlite(
    sqlite_conn: aiosqlite.Connection,
    experience_id: str,
    input_text: str,
    response: str,
    past_steps: List[Tuple[str, str]],
    errors: List[Dict[str, str]],
    has_error: bool,
    timestamp: str,
) -> bool:
    """写入 SQLite（结构化真相存储）

    Returns:
        是否写入成功
    """
    try:
        await sqlite_conn.execute(
            """
            INSERT INTO aiops_experiences
            (id, task, final_solution, steps_json, errors_json, task_type, has_error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experience_id,
                input_text,
                response,
                json.dumps(past_steps, ensure_ascii=False, default=str),
                json.dumps(errors, ensure_ascii=False, default=str),
                "aiops_diagnose",
                1 if has_error else 0,
                timestamp,
            ),
        )
        await sqlite_conn.commit()
        logger.info(f"经验已写入 SQLite: id={experience_id}")
        return True
    except Exception as e:
        logger.error(f"写入 SQLite 失败: {e}")
        return False


def make_memory_writer(sqlite_conn: aiosqlite.Connection | None):
    """创建经验回写节点（闭包绑定 SQLite 连接）

    Args:
        sqlite_conn: SQLite 异步连接，None 时仅写 Milvus

    Returns:
        LangGraph 节点函数
    """

    async def memory_writer(state: PlanExecuteState) -> Dict[str, Any]:
        """经验回写节点：把任务经验写入 LTM

        纯副作用节点，不修改 State。
        失败时记录日志并返回 {}，不阻塞主流程。
        """
        logger.info("=== Memory Writer：经验回写 ===")

        input_text = state.get("input", "")
        past_steps = state.get("past_steps", [])
        response = state.get("response", "")

        # 门控1：输入或响应为空，跳过
        if not input_text or not response:
            logger.warning("输入或响应为空，跳过经验回写")
            return {}

        try:
            # 1. 提取错误步骤
            errors = _extract_errors(past_steps)
            has_error = len(errors) > 0

            # 2. 构建经验摘要
            experience_text = _build_experience_text(
                input_text, response, errors, past_steps
            )
            experience_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            logger.info(
                f"经验回写: 任务长度={len(input_text)}, 方案长度={len(response)}, "
                f"步骤数={len(past_steps)}, 错误数={len(errors)}, has_error={has_error}"
            )

            # 3. 写入 Milvus（向量化检索）
            await _write_to_milvus(
                experience_id, experience_text, input_text, has_error, timestamp
            )

            # 4. 写入 SQLite（结构化真相）
            if sqlite_conn is not None:
                await _write_to_sqlite(
                    sqlite_conn,
                    experience_id,
                    input_text,
                    response,
                    past_steps,
                    errors,
                    has_error,
                    timestamp,
                )
            else:
                logger.warning("SQLite 连接为空，跳过结构化经验写入")

            return {}

        except Exception as e:
            logger.error(f"经验回写失败: {e}", exc_info=True)
            return {}  # 不阻塞流程

    return memory_writer


async def init_experiences_table(sqlite_conn: aiosqlite.Connection) -> None:
    """初始化经验表（在 AsyncSqliteSaver.setup 之后调用）

    Args:
        sqlite_conn: SQLite 异步连接
    """
    try:
        await sqlite_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aiops_experiences (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                final_solution TEXT,
                steps_json TEXT,
                errors_json TEXT,
                task_type TEXT,
                has_error INTEGER DEFAULT 0,
                created_at TEXT
            )
            """
        )
        # 创建时间索引，便于按时间范围查询/清理
        await sqlite_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aiops_experiences_created_at "
            "ON aiops_experiences(created_at)"
        )
        await sqlite_conn.commit()
        logger.info("aiops_experiences 表初始化完成")
    except Exception as e:
        logger.error(f"初始化 aiops_experiences 表失败: {e}")
        raise
