"""语义缓存服务 - 完整回答级缓存（P1）

缓存"问题向量 → 完整回答"映射，命中时跳过检索 + 重排 + LLM 生成。

存储：Milvus 独立 collection（semantic_cache），COSINE 度量。
失效双保险：
1. 知识库版本号（kb_version）—— 每次文档索引成功后递增，版本不一致即全量失效
2. TTL —— created_at 超过 semantic_cache_ttl_hours 自动过期

工程约束：
- 缓存任何故障（Milvus 不可用/超时）一律静默降级为不缓存，绝不影响主链路
- Milvus 连续失败后进入 5 分钟冷却期，避免每次请求都打一遍失败连接
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.config import config

# 知识库版本号文件（文档索引成功后递增，触发语义缓存全量失效）
KB_VERSION_PATH = Path("./data/kb_version.txt")
# Milvus 故障冷却期（秒）：冷却期内 lookup/store 直接旁路
_MILVUS_COOLDOWN_SECONDS = 300


class SemanticCacheService:
    """语义缓存服务 - Milvus collection + 知识库版本号 + TTL"""

    COLLECTION_NAME = "semantic_cache"
    VECTOR_DIM = 1024
    # Milvus varchar 上限（answer 截断保护）
    QUESTION_MAX_LENGTH = 4000
    ANSWER_MAX_LENGTH = 60000

    def __init__(self):
        self._collection = None
        self._cooldown_until = 0.0  # Milvus 故障冷却截止时间

    # ---------- 知识库版本号 ----------

    @staticmethod
    def get_kb_version() -> int:
        """读取当前知识库版本号（文件不存在返回 1）"""
        try:
            return int(KB_VERSION_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            return 1

    @staticmethod
    def bump_kb_version() -> int:
        """递增知识库版本号（文档索引成功后调用），语义缓存全量失效

        写入采用"临时文件 + 原子替换"，避免写一半被读到。
        """
        try:
            version = SemanticCacheService.get_kb_version() + 1
            KB_VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = KB_VERSION_PATH.with_suffix(".tmp")
            tmp.write_text(str(version), encoding="utf-8")
            tmp.replace(KB_VERSION_PATH)
            logger.info(f"知识库版本号递增: {version}（语义缓存全量失效）")
            return version
        except Exception as e:
            # 版本号写入失败不阻塞索引流程（缓存失效退化为依赖 TTL）
            logger.warning(f"知识库版本号递增失败: {e}")
            return 0

    # ---------- 缓存查询/写入 ----------

    def lookup(self, question: str) -> Optional[str]:
        """查询语义缓存

        Args:
            question: 用户问题

        Returns:
            Optional[str]: 命中返回缓存的完整回答；未命中/故障/关闭返回 None
        """
        if not config.semantic_cache_enabled:
            return None
        if time.time() < self._cooldown_until:
            return None

        try:
            from app.services.vector_embedding_service import vector_embedding_service

            self._ensure_collection()
            query_vector = vector_embedding_service.embed_query(question)

            # 失效条件：版本不一致（文档已更新）或超过 TTL
            cutoff = int(time.time()) - config.semantic_cache_ttl_hours * 3600
            expr = (
                f"kb_version == {self.get_kb_version()} "
                f"and created_at >= {cutoff}"
            )

            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=1,
                expr=expr,
                output_fields=["answer"],
            )

            for hits in results:
                for hit in hits:
                    # Milvus COSINE 度量：返回值即余弦相似度，越大越相似
                    if hit.distance >= config.semantic_cache_threshold:
                        answer = hit.entity.get("answer") or ""
                        if answer:
                            logger.info(
                                f"语义缓存命中: score={hit.distance:.4f}, "
                                f"answer长度={len(answer)}"
                            )
                            return answer

            logger.debug("语义缓存未命中")
            return None

        except Exception as e:
            self._enter_cooldown(e)
            return None

    def store(self, question: str, answer: str, sources: Optional[Dict[str, Any]] = None) -> None:
        """写入语义缓存（问题向量 → 完整回答）

        任何失败仅记录日志，不影响主链路。
        """
        if not config.semantic_cache_enabled:
            return
        if not question or not answer:
            return
        if time.time() < self._cooldown_until:
            return

        try:
            from app.services.vector_embedding_service import vector_embedding_service

            self._ensure_collection()
            vector = vector_embedding_service.embed_query(question)

            self._collection.insert(
                [
                    [str(uuid.uuid4())],                # id
                    [vector],                            # vector
                    [question[: self.QUESTION_MAX_LENGTH]],  # question
                    [answer[: self.ANSWER_MAX_LENGTH]],  # answer
                    [sources or {}],                     # sources (JSON)
                    [self.get_kb_version()],             # kb_version
                    [int(time.time())],                  # created_at
                ]
            )
            logger.info(
                f"语义缓存写入: question长度={len(question)}, answer长度={len(answer)}"
            )

        except Exception as e:
            self._enter_cooldown(e)

    # ---------- 内部：collection 管理 ----------

    def _ensure_collection(self) -> None:
        """确保缓存 collection 存在并已加载（幂等，复用 default 连接）"""
        if self._collection is not None:
            return

        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        from app.core.milvus_client import milvus_manager

        # 复用全局 Milvus 连接（幂等：已连接则跳过）
        milvus_manager.connect()

        if not utility.has_collection(self.COLLECTION_NAME):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(
                        name="id", dtype=DataType.VARCHAR,
                        max_length=100, is_primary=True,
                    ),
                    FieldSchema(
                        name="vector", dtype=DataType.FLOAT_VECTOR,
                        dim=self.VECTOR_DIM,
                    ),
                    FieldSchema(
                        name="question", dtype=DataType.VARCHAR,
                        max_length=self.QUESTION_MAX_LENGTH,
                    ),
                    FieldSchema(
                        name="answer", dtype=DataType.VARCHAR,
                        max_length=65535,
                    ),
                    FieldSchema(name="sources", dtype=DataType.JSON),
                    FieldSchema(name="kb_version", dtype=DataType.INT64),
                    FieldSchema(name="created_at", dtype=DataType.INT64),
                ],
                description="Semantic cache: question vector -> full answer",
                enable_dynamic_field=False,
            )
            collection = Collection(
                name=self.COLLECTION_NAME, schema=schema, num_shards=1
            )
            collection.create_index(
                field_name="vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            logger.info(f"语义缓存 collection 创建成功: {self.COLLECTION_NAME}")

        self._collection = Collection(self.COLLECTION_NAME)
        self._collection.load()

    def _enter_cooldown(self, error: Exception) -> None:
        """Milvus 故障进入冷却期（期间缓存完全旁路，避免拖慢每个请求）"""
        self._cooldown_until = time.time() + _MILVUS_COOLDOWN_SECONDS
        logger.warning(
            f"语义缓存故障，进入{_MILVUS_COOLDOWN_SECONDS}秒冷却期（缓存旁路）: {error}"
        )


# 全局单例
semantic_cache_service = SemanticCacheService()
