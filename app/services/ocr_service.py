"""OCR 服务模块 - 基于 DashScope qwen-vl 多模态模型的扫描页文字转写

流程：PyMuPDF 渲染 PDF 页面为 PNG → base64 → qwen-vl 转写纯文本。
失败/超时返回空串（调用方跳过该页，等同旧版无 OCR 行为），不阻塞整体解析。
"""

import base64
from typing import Optional

from loguru import logger
from openai import OpenAI

from app.config import config

# OCR 转写提示词：要求纯转写不解释，表格输出 Markdown（保留结构信息）
OCR_PROMPT = (
    "你是OCR转写助手。请转写图片中的所有文字内容，按从上到下、从左到右的阅读顺序"
    "输出纯文本，保持原有段落结构；如遇表格，请输出为Markdown表格格式；"
    "不要输出任何解释、注释或与图片内容无关的文字。"
)


class OcrService:
    """OCR 服务 - PyMuPDF 渲染 + qwen-vl 多模态转写"""

    def __init__(self):
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """延迟初始化 OpenAI 兼容客户端（复用 DashScope API Key）"""
        if self._client is None:
            self._client = OpenAI(
                api_key=config.dashscope_api_key,
                base_url=config.dashscope_api_base,
                timeout=config.ocr_timeout,
            )
        return self._client

    @staticmethod
    def render_page_png(pdf_path: str, page_index: int, dpi: int = 150) -> bytes:
        """渲染 PDF 指定页为 PNG 图片字节

        Args:
            pdf_path: PDF 文件路径
            page_index: 页索引（从 0 开始）
            dpi: 渲染分辨率（150dpi 对 A4 约 1240x1754，OCR 足够且体积可控）

        Returns:
            bytes: PNG 图片字节
        """
        import pymupdf

        doc = pymupdf.open(pdf_path)
        try:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(f"页索引越界: {page_index}（共 {len(doc)} 页）")
            pix = doc[page_index].get_pixmap(dpi=dpi)
            return pix.tobytes("png")
        finally:
            doc.close()

    def ocr_page(self, pdf_path: str, page_index: int) -> str:
        """OCR 转写 PDF 指定页（扫描页专用）

        Args:
            pdf_path: PDF 文件路径
            page_index: 页索引（从 0 开始）

        Returns:
            str: 转写文本；失败/超时/空结果返回空串（调用方跳过该页）
        """
        try:
            png_bytes = self.render_page_png(pdf_path, page_index)
            b64 = base64.b64encode(png_bytes).decode("utf-8")

            client = self._get_client()
            response = client.chat.completions.create(
                model=config.ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {"type": "text", "text": OCR_PROMPT},
                        ],
                    }
                ],
            )

            text = (response.choices[0].message.content or "").strip()
            if text:
                logger.info(
                    f"OCR 转写成功: {pdf_path} 第{page_index + 1}页, {len(text)} 字符"
                )
            else:
                logger.warning(f"OCR 转写返回空结果: {pdf_path} 第{page_index + 1}页")
            return text

        except Exception as e:
            # OCR 失败不阻塞解析：跳过该页（等同旧版无 OCR 行为）
            logger.warning(
                f"OCR 转写失败（跳过该页）: {pdf_path} 第{page_index + 1}页, 错误: {e}"
            )
            return ""


# 全局单例
ocr_service = OcrService()
