"""图片引用还原中间件 - 在「送给 LLM」的最后一刻把引用换成真图

为什么需要它：
消息里存的是轻量引用块（image_ref），模型不认识这种类型。必须在调模型前
换成标准的 image_url 块，同时**不能**让这次替换写回 state —— 否则 base64
又会被 checkpoint 快照下来，等于白做。

`wrap_model_call` 正好满足：它拿到的 ModelRequest 是调模型前的最后一站，
`request.override(messages=...)` 返回新实例、不改原对象，替换结果只作用于
这一次模型调用，不进 state、不进 checkpoint。

Token 控制：
只有「最近 N 轮」的图片会被还原成真图（N = chat_image_keep_recent_rounds，
默认 1，即仅当前轮）。更早的图片降级为一行文字占位，避免 keep 窗口内的图片
每轮重复计费视觉 token。
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware
from loguru import logger

from app.config import config
from app.services.image_store_service import IMAGE_REF_TYPE, image_store_service


def _is_image_ref(block: Any) -> bool:
    """判断 content 块是否为图片引用块"""
    return isinstance(block, dict) and block.get("type") == IMAGE_REF_TYPE


def _is_inline_image(block: Any) -> bool:
    """判断是否为内联 base64 图片块（旧会话遗留数据）

    方案上线前的会话里，图片是以 image_url + base64 dataURL 存的。
    这些历史消息同样要参与「超出窗口则降级」的处理，否则老会话继续每轮
    重复计费。
    """
    if not isinstance(block, dict) or block.get("type") != "image_url":
        return False
    url = block.get("image_url")
    if isinstance(url, dict):
        url = url.get("url", "")
    return isinstance(url, str) and url.startswith("data:image/")


def _placeholder(block: Dict[str, Any]) -> Dict[str, Any]:
    """把图片块降级为文字占位块

    保留「这里曾有一张图」的语义，让模型知道上文有图但不重复付费识别。
    """
    if _is_image_ref(block):
        size = block.get("bytes")
        hint = f"，约 {size} 字节" if isinstance(size, int) else ""
        text = f"[历史图片已省略{hint}；如需重新识别请再次发送该图片]"
    else:
        text = "[历史图片已省略；如需重新识别请再次发送该图片]"
    return {"type": "text", "text": text}


def _message_has_image(msg: Any) -> bool:
    """消息是否含图片块（引用或内联）"""
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return False
    return any(_is_image_ref(b) or _is_inline_image(b) for b in content)


def _rebuild_content(
    content: List[Any], hydrate: bool
) -> Tuple[List[Any], int, int]:
    """重建单条消息的 content

    Args:
        content: 原始 content 块列表
        hydrate: True 则把引用还原为真图；False 则降级为文字占位

    Returns:
        (新 content, 还原数, 降级数)
    """
    out: List[Any] = []
    restored = dropped = 0

    for block in content:
        if _is_image_ref(block):
            if not hydrate:
                out.append(_placeholder(block))
                dropped += 1
                continue
            data_url = image_store_service.load_data_url(block)
            if data_url is None:
                # 文件缺失/损坏：降级为占位，不让模型收到坏数据
                out.append(_placeholder(block))
                dropped += 1
            else:
                out.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
                restored += 1
        elif _is_inline_image(block) and not hydrate:
            # 旧会话的内联 base64：超出窗口同样降级
            out.append(_placeholder(block))
            dropped += 1
        else:
            out.append(block)

    return out, restored, dropped


class ImageHydrationMiddleware(AgentMiddleware):
    """把消息中的图片引用还原为模型可读的 image_url 块

    仅作用于模型调用这一瞬间，不影响 state / checkpoint 中存储的引用形式。
    """

    def _hydrate_messages(self, messages: List[Any]) -> Optional[List[Any]]:
        """构造供本次模型调用使用的消息列表

        Returns:
            新的消息列表；若无需改动则返回 None（省去一次拷贝）
        """
        # 找出含图消息的下标，只有最近 N 条含图消息保留真图
        image_indices = [i for i, m in enumerate(messages) if _message_has_image(m)]
        if not image_indices:
            return None

        keep_n = max(0, config.chat_image_keep_recent_rounds)
        keep_set = set(image_indices[-keep_n:]) if keep_n else set()
        # 预先建集合，避免在循环里反复构造（否则退化为 O(n²)）
        image_index_set = set(image_indices)

        new_messages: List[Any] = []
        changed = False
        total_restored = total_dropped = 0

        for i, msg in enumerate(messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, list) or i not in image_index_set:
                new_messages.append(msg)
                continue

            new_content, restored, dropped = _rebuild_content(
                content, hydrate=(i in keep_set)
            )
            total_restored += restored
            total_dropped += dropped

            if new_content == content:
                new_messages.append(msg)
                continue

            # 关键：copy 出新消息对象，不改原消息
            # （原消息属于 state，改了就会被 checkpoint 快照）
            try:
                new_messages.append(msg.model_copy(update={"content": new_content}))
            except AttributeError:
                # 非 pydantic 消息对象：退回浅拷贝
                import copy

                clone = copy.copy(msg)
                try:
                    clone.content = new_content
                    new_messages.append(clone)
                except Exception:
                    # 实在改不动就保持原样，宁可多花 token 也不能报错
                    new_messages.append(msg)
                    continue
            changed = True

        if not changed:
            return None

        if total_restored or total_dropped:
            logger.debug(
                f"图片引用处理: 还原 {total_restored} 张（送给模型）, "
                f"降级 {total_dropped} 张（文字占位）"
            )
        return new_messages

    def wrap_model_call(self, request, handler):
        """同步路径"""
        new_messages = self._hydrate_messages(request.messages)
        if new_messages is not None:
            request = request.override(messages=new_messages)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        """异步路径（本项目实际走这条）"""
        new_messages = self._hydrate_messages(request.messages)
        if new_messages is not None:
            request = request.override(messages=new_messages)
        return await handler(request)
