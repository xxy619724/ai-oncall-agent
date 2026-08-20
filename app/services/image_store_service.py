"""图片外置存储服务 - 让 base64 不进 checkpoint

背景：
LangGraph 每个 superstep 都会快照整个 messages 列表。若图片以 base64 dataURL
形式留在消息里，一张图会被复制 N 份（实测一张 27KB 的图在 4 个 checkpoint 中
放大成 114KB）。同时 keep 窗口内的消息每轮都全量重放给 LLM，视觉 token
被反复计费。

做法：
1. 收到 dataURL 时校验并落盘到 uploads/images/{session_id}/{sha256}.{ext}
2. 消息里只留轻量引用块 {"type": "image_ref", "path": ..., "sha256": ...}
3. 调 LLM 前由中间件把「最近一轮」的引用还原成真实 image_url 块
4. 会话清理时按 session_id 删除整个图片目录

安全：不信任前端声明的 MIME，按魔数（magic bytes）判定真实类型。
"""

import base64
import binascii
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from app.config import config

# 引用块类型标记（出现在消息 content 里，不含图片数据本身）
IMAGE_REF_TYPE = "image_ref"

# dataURL 前缀解析：data:<mime>;base64,<payload>
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<payload>.*)$",
    re.DOTALL,
)

# 魔数 → (规范 MIME, 扩展名)
# 只放行视觉模型确实支持、且不能承载脚本的位图格式。
# SVG 被刻意排除：它是 XML，可内嵌脚本，且视觉模型价值有限。
_MAGIC_SIGNATURES: Tuple[Tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
    (b"BM", "image/bmp", "bmp"),
)


class ImageValidationError(ValueError):
    """图片校验失败（大小超限 / 格式不支持 / base64 损坏）

    继承 ValueError 以兼容既有调用方的 except ValueError 分支。
    """


def _sniff_mime(raw: bytes) -> Optional[Tuple[str, str]]:
    """按魔数判定真实图片类型

    Args:
        raw: 解码后的图片字节

    Returns:
        (规范 MIME, 扩展名)，无法识别时返回 None
    """
    for magic, mime, ext in _MAGIC_SIGNATURES:
        if raw.startswith(magic):
            return mime, ext

    # WebP 需要同时校验 RIFF 容器头和 WEBP 标记（偏移 8）
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp", "webp"

    return None


class ImageStoreService:
    """图片落盘 + 引用还原"""

    def __init__(self, base_dir: str = ""):
        # 默认放在 uploads/images 下（uploads 已在 .gitignore 中）
        self._base = Path(base_dir) if base_dir else Path("./uploads/images")

    # ============================================================
    # 写入
    # ============================================================

    def store_data_url(self, session_id: str, data_url: str) -> Dict[str, Any]:
        """校验并落盘一张 dataURL 图片，返回可放入消息的引用块

        Args:
            session_id: 会话 ID（决定存储子目录，便于按会话清理）
            data_url: 完整 dataURL（data:image/png;base64,...）

        Returns:
            引用块 dict，形如
            {"type": "image_ref", "sha256": ..., "path": ..., "mime": ..., "bytes": ...}

        Raises:
            ImageValidationError: 格式非法 / 超出大小上限
        """
        data_url = (data_url or "").strip()
        if not data_url:
            raise ImageValidationError("图片内容为空")

        # 上限校验放在解码之前：避免先把超大 base64 解码进内存
        limit = config.chat_image_max_base64_size
        if len(data_url) > limit:
            raise ImageValidationError(
                f"图片过大: base64 长度 {len(data_url)} 超过上限 {limit}，请压缩后重试"
            )

        m = _DATA_URL_RE.match(data_url)
        if not m:
            raise ImageValidationError(
                "图片格式非法：需为 data:image/<type>;base64,<数据> 形式的 dataURL"
            )

        payload = m.group("payload").strip()
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ImageValidationError(f"图片 base64 解码失败: {e}") from e

        if not raw:
            raise ImageValidationError("图片解码后为空")

        # 不信任前端声明的 MIME，按魔数判定真实类型
        sniffed = _sniff_mime(raw)
        if sniffed is None:
            declared = m.group("mime")
            raise ImageValidationError(
                f"不支持的图片格式（声明为 {declared}，实际内容无法识别）。"
                f"仅支持 PNG / JPEG / GIF / BMP / WebP"
            )
        mime, ext = sniffed

        declared_mime = m.group("mime").lower()
        if declared_mime != mime:
            # 声明与实际不符：以实际为准并记录，可能是前端 bug 也可能是伪造尝试
            logger.warning(
                f"[会话 {session_id}] 图片声明类型 {declared_mime} 与实际内容 {mime} 不符，"
                f"按实际类型处理"
            )

        # 内容寻址：同一张图在同一会话内只存一份
        sha = hashlib.sha256(raw).hexdigest()
        target_dir = self._session_dir(session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{sha}.{ext}"

        if path.exists():
            logger.debug(f"[会话 {session_id}] 图片已存在，复用: {path.name}")
        else:
            # 先写临时文件再原子替换，避免并发/中断留下半张图
            tmp = path.with_suffix(path.suffix + ".part")
            try:
                tmp.write_bytes(raw)
                tmp.replace(path)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            logger.info(
                f"[会话 {session_id}] 图片已落盘: {path.name} "
                f"({len(raw)} 字节, {mime})"
            )

        return {
            "type": IMAGE_REF_TYPE,
            "sha256": sha,
            # 存相对路径：便于整个项目目录迁移
            "path": path.as_posix(),
            "mime": mime,
            "bytes": len(raw),
        }

    # ============================================================
    # 读取 / 还原
    # ============================================================

    def load_data_url(self, ref: Dict[str, Any]) -> Optional[str]:
        """把引用块还原成 dataURL（供调 LLM 前使用）

        Args:
            ref: store_data_url 返回的引用块

        Returns:
            dataURL 字符串；文件缺失或校验不符时返回 None（调用方降级为文字占位）
        """
        path_str = ref.get("path")
        if not path_str:
            return None

        path = Path(path_str)
        if not path.is_file():
            logger.warning(f"图片引用指向的文件不存在，降级为文字占位: {path}")
            return None

        try:
            raw = path.read_bytes()
        except OSError as e:
            logger.warning(f"图片读取失败，降级为文字占位: {path}, 错误: {e}")
            return None

        # 完整性校验：文件被替换/损坏时不把脏数据送给模型
        expected = ref.get("sha256")
        if expected:
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                logger.warning(
                    f"图片内容与引用记录的 sha256 不符，降级为文字占位: {path}"
                )
                return None

        mime = ref.get("mime") or "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # ============================================================
    # 清理
    # ============================================================

    def cleanup_session(self, session_id: str) -> int:
        """删除某会话的全部图片（会话清空 / checkpoint 过期时调用）

        Args:
            session_id: 会话 ID

        Returns:
            删除的文件数（目录不存在返回 0）
        """
        target = self._session_dir(session_id)
        if not target.is_dir():
            return 0

        count = sum(1 for p in target.iterdir() if p.is_file())
        try:
            shutil.rmtree(target)
            if count:
                logger.info(f"[会话 {session_id}] 已清理 {count} 张图片")
        except OSError as e:
            logger.warning(f"[会话 {session_id}] 图片清理失败: {target}, 错误: {e}")
            return 0
        return count

    def _session_dir(self, session_id: str) -> Path:
        """会话图片目录（session_id 经过消毒，防止穿越到其他路径）"""
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")
        # 纯点号的名字（. / ..）会指向父目录，单独兜底
        if safe.strip(".") == "":
            safe = "default"
        return self._base / safe


# 全局单例
image_store_service = ImageStoreService()
