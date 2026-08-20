"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    id: str = Field(..., description="会话 ID", alias="Id")
    question: str = Field(..., description="用户问题", alias="Question")
    image: str = Field(
        default="",
        description="可选图片，完整 dataURL 格式（data:image/png;base64,...），用于多模态识别",
        alias="Image",
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Id": "session-123",
                "Question": "这张图里的报错怎么解决？",
                "Image": "data:image/png;base64,iVBORw0KGgo..."
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True


class StopRequest(BaseModel):
    """终止流式输出请求"""

    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True
