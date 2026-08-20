"""文件上传接口模块"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import config
from app.services.vector_index_service import vector_index_service
from loguru import logger

router = APIRouter()

# 文件上传后存储的路径
UPLOAD_DIR = Path("./uploads")
# 支持的文件类型
ALLOWED_EXTENSIONS = ["txt", "md", "pdf", "docx"]
# 单个文件支持最大大小
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
# 分块读取上传流的块大小（避免一次性把整个请求体读进内存）
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


def _resolve_allowed_dir(directory_path: Optional[str]) -> Path:
    """把用户传入的目录解析为白名单内的绝对路径

    该端点会把目录下的文件内容读入向量库，之后可经 /api/chat 检索出来，
    因此必须限制在白名单目录内，否则可被用于读取宿主机任意路径。

    Args:
        directory_path: 用户传入的目录（相对项目根），None 表示默认 uploads

    Returns:
        校验通过的绝对路径

    Raises:
        HTTPException: 400 目录不在白名单内 / 不存在
    """
    project_root = Path.cwd().resolve()
    allowed_roots = [
        (project_root / d).resolve() for d in config.index_allowed_dirs_list
    ]

    if not directory_path:
        # 默认目录按需创建，与 /api/upload 的行为一致：
        # 全新部署尚未上传过任何文件时，应返回“索引 0 个文件”而不是 400
        target = (project_root / "uploads").resolve()
        target.mkdir(parents=True, exist_ok=True)
    else:
        # 先按相对项目根解析；绝对路径同样会被下面的白名单检查拦住
        candidate = Path(directory_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        target = candidate.resolve()

    # 必须落在某个白名单根目录之内（Path.is_relative_to 处理了 ../ 穿越）
    if not any(
        target == root or target.is_relative_to(root) for root in allowed_roots
    ):
        logger.warning(f"拒绝索引白名单外的目录: {directory_path!r} → {target}")
        raise HTTPException(
            status_code=400,
            detail=(
                f"目录不在允许范围内，仅支持: "
                f"{', '.join(config.index_allowed_dirs_list)}"
            ),
        )

    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {directory_path}")

    return target


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件并自动创建向量索引

    Args:
        file: 上传的文件

    Returns:
        JSONResponse: 上传结果
    """
    try:
        # 1. 验证文件
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        # 2. 规范化文件名（去除空格，处理 Windows 上传的文件）
        safe_filename = _sanitize_filename(file.filename)

        # 3. 验证文件扩展名
        file_extension = _get_file_extension(safe_filename)
        if file_extension not in ALLOWED_EXTENSIONS:
            # .doc 旧格式单独提示转换方式（P1：友好报错）
            if file_extension == "doc":
                raise HTTPException(
                    status_code=400,
                    detail="检测到 .doc 旧格式（Word 97-2003 二进制格式），暂不支持。"
                    "请用 Word 打开后另存为 .docx 再上传",
                )
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # 4. 创建上传目录
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        file_path = UPLOAD_DIR / safe_filename

        # 5. 先写临时文件并分块校验大小，全部成功后才替换旧文件
        # 这样超限上传不会破坏已有的同名文件，也不会把整个请求体读进内存
        tmp_path = file_path.with_suffix(file_path.suffix + ".part")
        total_size = 0
        try:
            with tmp_path.open("wb") as out:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=400,
                            detail=f"文件大小超过限制（最大 {MAX_FILE_SIZE} 字节）",
                        )
                    out.write(chunk)

            if total_size == 0:
                raise HTTPException(status_code=400, detail="文件内容为空")

            # 原子替换：os.replace 语义，旧文件仅在新文件完整落盘后被覆盖
            if file_path.exists():
                logger.info(f"文件已存在，将覆盖: {file_path}")
            tmp_path.replace(file_path)
        finally:
            # 校验失败或写入异常时清理残留的 .part
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        logger.info(f"文件上传成功: {file_path} ({total_size} 字节)")

        # 6. 自动创建向量索引
        try:
            logger.info(f"开始为上传文件创建向量索引: {file_path}")
            vector_index_service.index_single_file(str(file_path))
            logger.info(f"向量索引创建成功: {file_path}")
        except Exception as e:
            logger.error(f"向量索引创建失败: {file_path}, 错误: {e}")
            # 注意：即使索引失败，文件上传仍然成功，只是记录错误日志

        # 7. 返回响应
        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": total_size,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")


@router.post("/index_directory")
async def index_directory(directory_path: Optional[str] = None):
    """
    索引指定目录下的所有文件

    目录必须落在 INDEX_ALLOWED_DIRS 白名单内（默认 uploads、aiops-docs）。

    Args:
        directory_path: 目录路径（可选，相对项目根，默认使用 uploads 目录）

    Returns:
        JSONResponse: 索引结果
    """
    try:
        target_dir = _resolve_allowed_dir(directory_path)
        logger.info(f"开始索引目录: {target_dir}")

        # 执行索引
        result = vector_index_service.index_directory(str(target_dir))

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )

    except HTTPException:
        # 白名单校验产生的 400 需原样抛出，不能被下面的兜底转成 500
        raise
    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}")


def _get_file_extension(filename: str) -> str:
    """
    获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        str: 扩展名（小写，不含点）
    """
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].lower()
    return ""


def _sanitize_filename(filename: str) -> str:
    """
    规范化文件名，去除空格和特殊字符

    Args:
        filename: 原始文件名

    Returns:
        str: 规范化后的文件名
    """
    # 去除空格
    sanitized = filename.replace(" ", "_")
    # 去除其他可能导致问题的字符
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    return sanitized
