"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

from typing import Any, AsyncGenerator, Dict

import asyncio
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from loguru import logger
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS
from app.services.semantic_cache_service import semantic_cache_service
from app.services.image_store_service import (
    ImageValidationError,
    image_store_service,
)
from app.agent.image_hydration import ImageHydrationMiddleware
from app.agent.mcp_client import (
    get_mcp_client_with_retry,
    load_mcp_tools_safe,
    format_exception_chain,
    suggest_mcp_transport,
)

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        # 聊天链路使用多模态模型（qwen-vl-max）：支持用户发送图片，
        # 纯文本请求完全兼容，工具调用（function calling）同样支持
        self.model_name = config.chat_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0.7,
            streaming=streaming,
            profile={"max_input_tokens": config.rag_context_window_size},
        )

        # 定义基础工具（与 AIOps Planner/Executor 使用同一套默认本地工具）
        self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 临时使用 MemorySaver，在 _initialize_agent 中替换为 AsyncSqliteSaver
        self.checkpointer = MemorySaver()
        self._sqlite_conn = None
        self._cleaner = None  # SQLite Checkpoint 清理器

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        # 主动终止注册表：session_id → 停止标记（流式输出中逐 token 检查）
        self._stop_flags: Dict[str, asyncio.Event] = {}

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具和 SummarizationMiddleware）"""
        if self._agent_initialized:
            return

        for name, server in config.mcp_servers.items():
            hint = suggest_mcp_transport(
                str(server.get("url", "")),
                str(server.get("transport", "")),
            )
            if hint:
                logger.warning(f"MCP 配置 [{name}]: {hint}")

        mcp_client = await get_mcp_client_with_retry()
        mcp_tools, mcp_err = await load_mcp_tools_safe(mcp_client)
        if mcp_err:
            logger.warning(
                f"MCP 工具加载失败，将仅使用本地工具继续运行:\n{mcp_err}"
            )
            self.mcp_tools = []
        else:
            self.mcp_tools = mcp_tools
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        all_tools = self.tools + self.mcp_tools

        # 初始化 AsyncSqliteSaver（异步持久化检查点），失败降级为 MemorySaver
        try:
            self._sqlite_conn = await aiosqlite.connect(config.sqlite_db_path)
            self.checkpointer = AsyncSqliteSaver(self._sqlite_conn)
            await self.checkpointer.setup()
            logger.info(f"AsyncSqliteSaver 初始化成功: {config.sqlite_db_path}")

            # 初始化 Checkpoint 清理器并启动定时任务
            from app.services.checkpoint_cleaner import SqliteCheckpointCleaner
            self._cleaner = SqliteCheckpointCleaner(self._sqlite_conn)
            await self._cleaner.start_periodic_cleanup()

            # 初始化经验 TTL 清理器并启动定时任务（软删除过期经验）
            from app.services.experience_ttl_cleaner import ExperienceTtlCleaner
            self._ttl_cleaner = ExperienceTtlCleaner(self._sqlite_conn)
            await self._ttl_cleaner.start_periodic_cleanup()
        except Exception as e:
            logger.error(f"AsyncSqliteSaver 初始化失败，降级为 MemorySaver: {e}")
            self._sqlite_conn = None
            self._cleaner = None
            self._ttl_cleaner = None
            self.checkpointer = MemorySaver()

        # 上下文自动压缩中间件：两层记忆架构
        # 第一层（滑动窗口）：保留最近 N 轮完整对话 → keep
        # 第二层（摘要记忆）：超出窗口时自动 LLM 压缩旧消息 → trigger
        summarization_middleware = SummarizationMiddleware(
            model=self.model,
            trigger=[
                ("fraction", config.rag_compression_threshold),
                ("messages", config.rag_keep_recent_rounds * 2 + 4),
            ],
            keep=("messages", config.rag_keep_recent_rounds * 2),
        )

        # 图片引用还原中间件：调模型前把 image_ref 换成真实 image_url。
        # 必须排在压缩中间件之后（即更靠内层）：压缩看到的是轻量引用块，
        # 不会把 base64 计入 token 预算；还原发生在最后一刻，不写回 state。
        middleware = [summarization_middleware]
        if config.chat_image_external_store:
            middleware.append(ImageHydrationMiddleware())

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
            system_prompt=self.system_prompt,
            middleware=middleware,
        )

        self._agent_initialized = True

        # 初始化 MemoryService（双层记忆服务）
        from app.services.memory_service import init_memory_service
        init_memory_service(self.checkpointer)

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    @staticmethod
    def _build_message_content(
        question: str, image_data_url: str = "", session_id: str = "default"
    ) -> Any:
        """构建消息内容：无图时返回纯文本，带图时返回多模态 content 块列表

        带图时默认走外置存储：图片落盘，消息里只留轻量引用块（image_ref），
        真图由 ImageHydrationMiddleware 在调模型前还原。这样 base64 不会被
        checkpoint 快照反复复制（实测一张 27KB 图曾放大成 114KB）。

        Args:
            question: 用户问题文本
            image_data_url: 可选图片，完整 dataURL（data:image/png;base64,...）
            session_id: 会话 ID（决定图片存储目录，便于按会话清理）

        Returns:
            str 或 list: LangChain 消息 content（多模态时为 content 块列表）

        Raises:
            ImageValidationError: 图片超限 / 格式非法（ValueError 子类）
        """
        image_data_url = (image_data_url or "").strip()

        if not image_data_url:
            return question

        text_block = {"type": "text", "text": question or "请识别这张图片"}

        if config.chat_image_external_store:
            # 落盘 + 引用（大小与魔数校验都在 store_data_url 内完成）
            ref = image_store_service.store_data_url(session_id, image_data_url)
            return [text_block, ref]

        # 兼容开关关闭时的旧行为：base64 直接进消息
        if len(image_data_url) > config.chat_image_max_base64_size:
            raise ValueError(
                f"图片过大: base64 长度 {len(image_data_url)} 超过上限 "
                f"{config.chat_image_max_base64_size}，请压缩后重试"
            )
        return [
            text_block,
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]

    async def query(
        self,
        question: str,
        session_id: str,
        image_data_url: str = "",
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            image_data_url: 可选图片（dataURL 格式），用于多模态识别

        Returns:
            str: 完整答案
        """
        try:
            # P1 语义缓存：命中直接返回完整回答（跳过检索 + 生成）
            # 图片问题不缓存（视觉识别结果时效性差、复用价值低）
            if config.semantic_cache_enabled and not image_data_url:
                cached = await asyncio.to_thread(
                    semantic_cache_service.lookup, question
                )
                if cached is not None:
                    logger.info(f"[会话 {session_id}] 语义缓存命中，直接返回")
                    return cached

            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # system prompt 由 agent 托管（create_agent 时注入），无需重复传入
            content = self._build_message_content(
                question, image_data_url, session_id=session_id
            )
            messages = [HumanMessage(content=content)]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")

                # 同步摘要到 Redis
                from app.services.memory_service import get_memory_service
                mem_service = get_memory_service()
                if mem_service:
                    await mem_service.sync_summary(session_id)

                # 更新会话活跃时间（供清理服务判断过期）
                if self._cleaner:
                    await self._cleaner.touch_session(
                        session_id, message_count=len(messages_result)
                    )

                # P1 语义缓存：写入完整回答（失败内部静默降级，不影响返回）
                if config.semantic_cache_enabled and not image_data_url and answer:
                    await asyncio.to_thread(
                        semantic_cache_service.store, question, str(answer)
                    )

                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（非流式）: "
                f"{format_exception_chain(e)}"
            )
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
        image_data_url: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            image_data_url: 可选图片（dataURL 格式），用于多模态识别

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "stopped" | "error"
                - data: 具体内容
        """
        # 注册停止标记（新事件覆盖旧条目，天然忽略历史残留的停止请求）
        stop_event = asyncio.Event()
        self._stop_flags[session_id] = stop_event

        try:
            # P1 语义缓存：命中直接返回完整回答（一次性推送给前端）
            if config.semantic_cache_enabled and not image_data_url:
                cached = await asyncio.to_thread(
                    semantic_cache_service.lookup, question
                )
                if cached is not None:
                    logger.info(f"[会话 {session_id}] 语义缓存命中（流式）")
                    yield {"type": "content", "data": cached}
                    yield {"type": "complete"}
                    return

            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # system prompt 由 agent 托管（create_agent 时注入），无需重复传入
            content = self._build_message_content(
                question, image_data_url, session_id=session_id
            )
            messages = [HumanMessage(content=content)]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            stopped = False
            full_answer = ""  # 累计完整回答（语义缓存写入用）
            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                # 主动终止：token 边界检查停止标记，命中即中断流式输出
                if stop_event.is_set():
                    stopped = True
                    break

                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    full_answer += text_content
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            if stopped:
                # 用户主动终止：跳过摘要同步与缓存写入（本轮回答未完成，不入 checkpoint），
                # 已生成的部分内容已通过 content 事件推给前端
                logger.info(f"[会话 {session_id}] 收到停止请求，流式输出已终止")
                yield {"type": "stopped"}
                return

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")

            # 同步摘要到 Redis
            from app.services.memory_service import get_memory_service
            mem_service = get_memory_service()
            if mem_service:
                await mem_service.sync_summary(session_id)

            # 更新会话活跃时间（供清理服务判断过期）
            if self._cleaner:
                await self._cleaner.touch_session(session_id)

            # P1 语义缓存：写入完整回答（失败内部静默降级）
            if config.semantic_cache_enabled and not image_data_url and full_answer:
                await asyncio.to_thread(
                    semantic_cache_service.store, question, full_answer
                )

            yield {"type": "complete"}

        except ImageValidationError:
            # 图片校验失败属客户端输入问题，交给 API 层统一回 400。
            # 此处安全：校验发生在任何 yield 之前（带图时会跳过语义缓存分支），
            # 不存在「已推送部分内容又抛异常」的情况。
            raise

        except Exception as e:
            detail = format_exception_chain(e)
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（流式）: {detail}"
            )
            yield {"type": "error", "data": detail}

        finally:
            # 流结束（正常/停止/异常）后清理注册表，防止内存泄漏
            self._stop_flags.pop(session_id, None)

    def request_stop(self, session_id: str) -> bool:
        """请求终止指定会话的流式输出（由 /chat/stop 调用）

        Args:
            session_id: 会话 ID

        Returns:
            bool: True 表示找到运行中的流式任务并已设置停止标记；
                  False 表示该会话当前没有正在进行的流式输出
        """
        stop_event = self._stop_flags.get(session_id)
        if stop_event is None:
            return False
        stop_event.set()
        logger.info(f"[会话 {session_id}] 收到主动终止请求")
        return True

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """提取消息文本：多模态 content（list 块）只取文本块，防止 base64 泄漏到前端

        Args:
            content: 消息 content（str 或多模态 content 块 list）

        Returns:
            str: 纯文本内容
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return " ".join(p for p in parts if p)
        return str(content)

    async def aget_session_history(self, session_id: str) -> list:
        """获取会话历史（委托给 MemoryService）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        # 确保 checkpointer 已初始化为 AsyncSqliteSaver，MemoryService 已就绪
        await self._initialize_agent()

        from app.services.memory_service import get_memory_service

        mem_service = get_memory_service()
        if mem_service:
            return await mem_service.aget_history(session_id)

        # 降级：直接从 checkpointer 异步读取
        try:
            config_dict = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = await self.checkpointer.aget_tuple(config_dict)

            if not checkpoint_tuple:
                return []

            checkpoint_data = (
                checkpoint_tuple.checkpoint
                if hasattr(checkpoint_tuple, 'checkpoint')
                else {}
            )

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                # 多模态消息 content 为 list 块，只提取文本块（防止 base64 泄漏到前端）
                content = self._extract_text_content(
                    msg.content if hasattr(msg, 'content') else str(msg)
                )
                timestamp = getattr(msg, 'timestamp', None)
                if not timestamp:
                    from datetime import datetime
                    timestamp = datetime.now().isoformat()
                history.append({"role": role, "content": content, "timestamp": timestamp})

            logger.info(f"获取会话历史(降级): {session_id}, 消息数量: {len(history)}")
            return history
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    async def aclear_session(self, session_id: str) -> bool:
        """清空会话历史（委托给 MemoryService）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        # 确保 checkpointer 已初始化为 AsyncSqliteSaver，MemoryService 已就绪
        await self._initialize_agent()

        # 同步清理该会话的外置图片，避免消息删了、图片文件却永久残留
        # （放在删消息之前：即使后续删除失败，图片也不会变成无主文件）
        try:
            await asyncio.to_thread(
                image_store_service.cleanup_session, session_id
            )
        except Exception as e:
            # 图片清理失败不应阻断会话清空这个主流程
            logger.warning(f"[会话 {session_id}] 清理图片失败（继续清空会话）: {e}")

        from app.services.memory_service import get_memory_service

        mem_service = get_memory_service()
        if mem_service:
            return await mem_service.aclear(session_id)

        # 降级：直接从 checkpointer 异步删除
        try:
            await self.checkpointer.adelete_thread(session_id)
            logger.info(f"已清除会话历史(降级): {session_id}")
            return True
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")

            # 停止 Checkpoint 清理器
            if self._cleaner:
                await self._cleaner.stop()

            # 停止经验 TTL 清理器
            if hasattr(self, "_ttl_cleaner") and self._ttl_cleaner:
                await self._ttl_cleaner.stop()

            # 关闭 MemoryService Redis 连接
            from app.services.memory_service import get_memory_service
            mem_service = get_memory_service()
            if mem_service:
                mem_service.close_redis()

            # 关闭 SQLite 连接
            if hasattr(self, "_sqlite_conn") and self._sqlite_conn:
                await self._sqlite_conn.close()
                logger.info("SQLite 连接已关闭")

            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
