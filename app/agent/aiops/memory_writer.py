"""经验回写节点：任务完成后将经验结构化写入 LTM（Milvus + SQLite）

对应记忆工程文档信息流的最后一环：判断是否需要写入 Memory。
- 写入门控：查重 → 冲突检测 → 置信度评估 → 写入
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

from app.config import config
from .state import PlanExecuteState
from app.observability import trace_node


# 错误识别关键词（用于从 past_steps 中提取失败步骤）
_ERROR_KEYWORDS = ("失败", "错误", "异常", "error", "failed", "exception", "traceback")

# 外部工具关键词（用于识别低置信度来源）
_EXTERNAL_TOOL_KEYWORDS = ("mcp", "外部", "第三方", "接口调用", "api")


# ============================================================
# 第一部分：基础工具函数（提取错误、构建经验文本）
# ============================================================

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


# ============================================================
# 第二部分：写入门控（查重、冲突检测、置信度评估）
# ============================================================

def _assess_confidence(
    past_steps: List[Tuple[str, str]], response: str
) -> str:
    """评估经验置信度

    规则：
    - past_steps 含外部工具返回（MCP/第三方）→ low（外部内容不可全信）
    - past_steps 含工具调用成功结果（非错误、非外部）→ high（工具数据可信）
    - 纯模型推理（response 无工具支撑）→ medium

    Returns:
        "high" / "medium" / "low"
    """
    has_tool_success = False
    has_external = False

    for step, result in past_steps:
        result_str = result if isinstance(result, str) else str(result)
        result_lower = result_str.lower()

        # 检查是否含外部工具关键词
        if any(kw in result_lower for kw in _EXTERNAL_TOOL_KEYWORDS):
            has_external = True
            break  # 外部来源直接 low，不用继续

        # 检查是否含工具调用成功结果（非错误）
        if not any(kw in result_lower for kw in _ERROR_KEYWORDS):
            has_tool_success = True

    if has_external:
        return "low"
    elif has_tool_success:
        return "high"
    else:
        return "medium"


async def _check_duplicate(
    input_text: str, dedup_threshold: float
) -> Tuple[bool, List[Tuple[Any, float]]]:
    """查重：检索是否已存在相似经验

    流程：
    1. 用 input_text 在 Milvus 中相似度检索
    2. 过滤只看经验类文档（_source == aiops_experience）
    3. 相似度 ≥ dedup_threshold → 判定重复，跳过写入
    4. 相似度 ≥ conflict_threshold → 返回相似文档，进入冲突检测
    5. 相似度 < conflict_threshold → 无相似，正常写入

    Returns:
        (is_duplicate, similar_docs_with_scores)
        - is_duplicate: True 表示重复，应跳过写入
        - similar_docs_with_scores: 相似文档列表 [(doc, score), ...]
    """
    try:
        from app.services.vector_store_manager import vector_store_manager

        # 用 similarity_search_with_score 获取相似度分数
        # Milvus COSINE 的 score 范围是 [0, 1]，越接近 1 越相似
        docs_with_scores = vector_store_manager.vector_store.similarity_search_with_score(
            input_text, k=3, param={"metric_type": "COSINE", "params": {"nprobe": 10}}
        )

        # 过滤：只看经验类文档（不看知识库文档）
        exp_with_scores = [
            (doc, score)
            for doc, score in docs_with_scores
            if doc.metadata.get("_source") == "aiops_experience"
        ]

        if not exp_with_scores:
            return False, []

        # 取最高分
        top_doc, top_score = exp_with_scores[0]
        logger.info(f"查重: 最高相似度={top_score:.4f}, 经验文档数={len(exp_with_scores)}")

        # 相似度 ≥ 查重阈值 → 重复，跳过
        if top_score >= dedup_threshold:
            logger.info(f"查重命中: {top_score:.4f} >= {dedup_threshold}, 跳过写入")
            return True, exp_with_scores

        # 相似度 ≥ 冲突阈值 → 进入冲突检测
        if top_score >= config.memory_conflict_threshold:
            logger.info(f"查重: {top_score:.4f} >= {config.memory_conflict_threshold}, 进入冲突检测")
            return False, exp_with_scores

        # 相似度低，正常写入
        return False, []

    except Exception as e:
        logger.error(f"查重失败: {e}")
        return False, []  # 查重失败不阻塞写入


async def _detect_conflict(
    new_input: str,
    new_response: str,
    new_has_error: bool,
    old_doc: Any,
) -> Tuple[bool, str]:
    """冲突检测：新经验与旧经验是否矛盾

    规则：
    1. has_error 不同（旧无错误但新有错误，或反之）→ 冲突
    2. response 关键词矛盾（如旧说"正常"新说"异常"）→ 冲突

    Returns:
        (is_conflict, reason)
    """
    old_metadata = old_doc.metadata
    old_has_error = old_metadata.get("has_error", False)

    # 规则1：has_error 不同
    if old_has_error != new_has_error:
        return True, f"has_error 不同: 旧={old_has_error}, 新={new_has_error}"

    # 规则2：response 关键词矛盾
    old_preview = old_metadata.get("task_preview", "")
    contradiction_pairs = [
        ("正常", "异常"),
        ("成功", "失败"),
        ("正常", "错误"),
        ("通过", "拒绝"),
    ]
    for pos, neg in contradiction_pairs:
        if pos in old_preview and neg in new_response:
            return True, f"response 矛盾: 旧含'{pos}', 新含'{neg}'"
        if neg in old_preview and pos in new_response:
            return True, f"response 矛盾: 旧含'{neg}', 新含'{pos}'"

    return False, ""


async def _mark_old_deprecated(
    sqlite_conn: aiosqlite.Connection | None, old_doc: Any
) -> None:
    """标记旧经验为 deprecated（软删除）

    在 SQLite 中更新 status=deprecated，不物理删除，保留审计。
    Milvus 中的旧向量暂时保留（下次检索时可通过 status 过滤）。
    """
    old_id = old_doc.metadata.get("experience_id")
    if not old_id:
        logger.warning("旧经验无 experience_id, 无法标记 deprecated")
        return

    if sqlite_conn is None:
        logger.warning("SQLite 连接为空, 跳过 deprecated 标记")
        return

    try:
        await sqlite_conn.execute(
            "UPDATE aiops_experiences SET status = 'deprecated' WHERE id = ?",
            (old_id,),
        )
        await sqlite_conn.commit()
        logger.info(f"旧经验已标记 deprecated: id={old_id}")
    except Exception as e:
        logger.error(f"标记旧经验 deprecated 失败: {e}")


# ============================================================
# 第三部分：存储写入（Milvus + SQLite）
# ============================================================

async def _write_to_milvus(
    experience_id: str,
    experience_text: str,
    input_text: str,
    has_error: bool,
    timestamp: str,
    confidence: str,
    status: str,
    ttl_days: int,
    version: int,
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
                # ===== 新增元数据（支撑门控与遗忘） =====
                "confidence": confidence,   # 置信度: high/medium/low
                "status": status,           # 状态: active/pending/deprecated
                "ttl_days": ttl_days,       # TTL（天），超时可软删除
                "version": version,         # 版本号，冲突时旧版+1
            },
        )
        vector_store_manager.add_documents([doc])
        logger.info(
            f"经验已写入 Milvus: id={experience_id}, "
            f"confidence={confidence}, status={status}"
        )
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
    confidence: str,
    status: str,
    ttl_days: int,
    version: int,
) -> bool:
    """写入 SQLite（结构化真相存储）

    Returns:
        是否写入成功
    """
    try:
        await sqlite_conn.execute(
            """
            INSERT INTO aiops_experiences
            (id, task, final_solution, steps_json, errors_json, task_type, has_error, created_at,
             confidence, status, ttl_days, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                confidence,
                status,
                ttl_days,
                version,
            ),
        )
        await sqlite_conn.commit()
        logger.info(
            f"经验已写入 SQLite: id={experience_id}, "
            f"confidence={confidence}, status={status}"
        )
        return True
    except Exception as e:
        logger.error(f"写入 SQLite 失败: {e}")
        return False


# ============================================================
# 第四部分：记忆回写节点（门控整合）
# ============================================================

def make_memory_writer(sqlite_conn: aiosqlite.Connection | None):
    """创建经验回写节点（闭包绑定 SQLite 连接）

    Args:
        sqlite_conn: SQLite 异步连接，None 时仅写 Milvus

    Returns:
        LangGraph 节点函数
    """

    @trace_node("memory_writer")
    async def memory_writer(state: PlanExecuteState) -> Dict[str, Any]:
        """经验回写节点：把任务经验写入 LTM

        门控流程：查重 → 冲突检测 → 置信度评估 → 写入
        纯副作用节点，不修改 State。
        失败时记录日志并返回 {}，不阻塞主流程。
        """
        logger.info("=== Memory Writer：经验回写（含门控）===")

        input_text = state.get("input", "")
        past_steps = state.get("past_steps", [])
        response = state.get("response", "")

        # 门控1：输入或响应为空，跳过
        if not input_text or not response:
            logger.warning("输入或响应为空，跳过经验回写")
            return {}

        try:
            # ===== 门控2：写入前查重 =====
            is_duplicate, similar_docs = await _check_duplicate(
                input_text, config.memory_dedup_threshold
            )

            if is_duplicate:
                logger.info("查重命中：已存在高度相似经验，跳过写入")
                return {}

            # ===== 门控3：冲突检测 =====
            # 对相似但非重复的经验，检查是否矛盾
            if similar_docs:
                new_has_error = len(_extract_errors(past_steps)) > 0
                for old_doc, score in similar_docs:
                    is_conflict, reason = await _detect_conflict(
                        input_text, response, new_has_error, old_doc
                    )
                    if is_conflict:
                        logger.info(f"冲突检测：与旧经验冲突（{reason}），标记旧经验 deprecated")
                        await _mark_old_deprecated(sqlite_conn, old_doc)
                        # 冲突时标记旧经验后，继续写入新经验（不 break，可能有多条冲突）

            # ===== 门控4：置信度评估 =====
            confidence = _assess_confidence(past_steps, response)
            # 低置信度标记 pending（待确认），不立即激活
            status = "pending" if confidence == "low" else "active"

            if confidence == "low":
                logger.info("置信度评估：low（外部工具来源），标记 status=pending（待确认）")
            elif confidence == "high":
                logger.info("置信度评估：high（工具数据支撑）")
            else:
                logger.info("置信度评估：medium（模型推理）")

            # ===== 提取错误步骤 =====
            errors = _extract_errors(past_steps)
            has_error = len(errors) > 0

            # ===== 构建经验摘要 =====
            experience_text = _build_experience_text(
                input_text, response, errors, past_steps
            )
            experience_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            ttl_days = config.memory_default_ttl_days
            version = 1

            logger.info(
                f"经验回写: 任务长度={len(input_text)}, 方案长度={len(response)}, "
                f"步骤数={len(past_steps)}, 错误数={len(errors)}, "
                f"has_error={has_error}, confidence={confidence}, status={status}"
            )

            # ===== 写入 Milvus（向量化检索）=====
            await _write_to_milvus(
                experience_id, experience_text, input_text, has_error, timestamp,
                confidence, status, ttl_days, version,
            )

            # ===== 写入 SQLite（结构化真相）=====
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
                    confidence,
                    status,
                    ttl_days,
                    version,
                )
            else:
                logger.warning("SQLite 连接为空，跳过结构化经验写入")

            return {}

        except Exception as e:
            logger.error(f"经验回写失败: {e}", exc_info=True)
            return {}  # 不阻塞流程

    return memory_writer


# ============================================================
# 第五部分：表初始化
# ============================================================

async def init_experiences_table(sqlite_conn: aiosqlite.Connection) -> None:
    """初始化经验表（在 AsyncSqliteSaver.setup 之后调用）

    表结构含门控字段：confidence / status / ttl_days / version
    兼容旧表：自动 ALTER TABLE 添加新列。

    Args:
        sqlite_conn: SQLite 异步连接
    """
    try:
        # 1. 创建表（不存在时）
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
                created_at TEXT,
                confidence TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'active',
                ttl_days INTEGER DEFAULT 90,
                version INTEGER DEFAULT 1
            )
            """
        )

        # 2. 兼容旧表：如果表已存在但缺新列，用 ALTER TABLE 添加
        # SQLite 的 ALTER TABLE ADD COLUMN 不支持 IF NOT EXISTS，用 try-except
        new_columns = [
            ("confidence", "TEXT DEFAULT 'medium'"),
            ("status", "TEXT DEFAULT 'active'"),
            ("ttl_days", "INTEGER DEFAULT 90"),
            ("version", "INTEGER DEFAULT 1"),
        ]
        for col_name, col_def in new_columns:
            try:
                await sqlite_conn.execute(
                    f"ALTER TABLE aiops_experiences ADD COLUMN {col_name} {col_def}"
                )
                logger.info(f"已添加列: {col_name} ({col_def})")
            except Exception:
                # 列已存在，正常情况，忽略
                pass

        # 3. 创建索引
        await sqlite_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aiops_experiences_created_at "
            "ON aiops_experiences(created_at)"
        )
        await sqlite_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aiops_experiences_status "
            "ON aiops_experiences(status)"
        )
        await sqlite_conn.commit()
        logger.info("aiops_experiences 表初始化完成（含门控字段：confidence/status/ttl_days/version）")
    except Exception as e:
        logger.error(f"初始化 aiops_experiences 表失败: {e}")
        raise
