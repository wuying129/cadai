#!/usr/bin/env python3
"""
AI电商海报生成系统 - FastAPI后端服务
提供完整的API接口，包含日志记录和数据持久化
"""

import os
import sys
import json
import uuid
import base64
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 导入核心模块
from backend.core.design_generator import DesignGenerator
from backend.core.poster_generator import PosterGenerator
from backend.core.batch_processor import BatchProcessor
from backend.core.poster_editor import PosterEditor
from backend.utils.logger import setup_logger, TaskLogger
from backend.utils.storage import StorageManager
from backend.utils.version_manager import VersionManager
from backend.models.edit_models import EditRequest, EditTaskStatus, EditTaskResponse

# ==================== 配置 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

# 确保目录存在
for dir_path in [DATA_DIR, LOGS_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 设置日志
logger = setup_logger("api", LOGS_DIR / "api.log")

# ==================== 数据模型 ====================

class TaskStatus(BaseModel):
    task_id: str
    user_id: str
    product_id: str
    status: str  # pending, processing, design_complete, generating, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    updated_at: str
    design_output: Optional[str] = None
    prompts: Optional[dict] = None
    posters: Optional[List[dict]] = None
    error: Optional[str] = None


class GenerateRequest(BaseModel):
    user_id: str
    product_id: str
    product_description: str


class TaskResponse(BaseModel):
    success: bool
    task_id: Optional[str] = None
    message: str


# ==================== 任务管理 ====================

# 内存中的任务状态（生产环境应使用Redis或数据库）
tasks_db: dict[str, TaskStatus] = {}
edit_tasks_db: dict[str, EditTaskStatus] = {}
version_manager = VersionManager(DATA_DIR)
storage = StorageManager(DATA_DIR)


def create_task(user_id: str, product_id: str) -> TaskStatus:
    """创建新任务"""
    task_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    task = TaskStatus(
        task_id=task_id,
        user_id=user_id,
        product_id=product_id,
        status="pending",
        progress=0,
        message="任务已创建，等待处理",
        created_at=now,
        updated_at=now
    )

    tasks_db[task_id] = task
    logger.info(f"创建任务: {task_id} | 用户: {user_id} | 产品: {product_id}")

    return task


def update_task(task_id: str, **kwargs):
    """更新任务状态"""
    if task_id not in tasks_db:
        return

    task = tasks_db[task_id]
    for key, value in kwargs.items():
        if hasattr(task, key):
            setattr(task, key, value)

    task.updated_at = datetime.now().isoformat()
    logger.info(f"更新任务: {task_id} | 状态: {task.status} | 进度: {task.progress}%")


# ==================== 后台任务处理 ====================

async def process_generation_task(
    task_id: str,
    user_id: str,
    product_id: str,
    product_description: str,
    image_paths: List[str]
):
    """后台执行完整的海报生成流程"""

    # 创建任务专属日志器
    task_logger = TaskLogger(task_id, LOGS_DIR)

    try:
        task_logger.log("=" * 60)
        task_logger.log(f"🚀 开始处理任务: {task_id}")
        task_logger.log(f"   用户ID: {user_id}")
        task_logger.log(f"   产品ID: {product_id}")
        task_logger.log(f"   产品描述: {product_description[:100]}...")
        task_logger.log(f"   图片数量: {len(image_paths)}")
        task_logger.log("=" * 60)

        # ========== 阶段1: 设计AI生成提示词 ==========
        update_task(task_id, status="processing", progress=10, message="AI正在分析产品...")
        task_logger.log("\n📝 阶段1: 调用设计AI生成提示词")

        design_gen = DesignGenerator()
        design_result = await design_gen.generate(
            text_input=product_description,
            image_paths=image_paths
        )

        if not design_result.get("success"):
            raise Exception(f"设计AI生成失败: {design_result.get('error', '未知错误')}")

        design_output = design_result.get("raw_response", "")

        # 保存设计输出
        design_path = storage.save_design_output(user_id, product_id, design_output)
        task_logger.log(f"   ✓ 设计输出已保存: {design_path}")
        task_logger.log(f"   ✓ 输出长度: {len(design_output)} 字符")

        update_task(
            task_id,
            status="design_complete",
            progress=30,
            message="设计方案生成完成，正在解析...",
            design_output=design_output
        )

        # ========== 阶段2: 解析提示词 ==========
        task_logger.log("\n🔍 阶段2: 解析设计输出")

        batch_processor = BatchProcessor()
        prompts = batch_processor.parse_design_output(design_output)

        if not prompts:
            raise Exception("解析失败：未找到有效的海报提示词")

        # 保存提示词JSON
        prompts_path = storage.save_prompts(user_id, product_id, prompts)
        task_logger.log(f"   ✓ 找到 {len(prompts)} 个海报提示词")
        task_logger.log(f"   ✓ 提示词已保存: {prompts_path}")

        for name in prompts.keys():
            task_logger.log(f"      - {name}")

        update_task(
            task_id,
            status="generating",
            progress=40,
            message=f"开始生成 {len(prompts)} 张海报...",
            prompts=prompts
        )

        # ========== 阶段3: 批量生成海报图片（并发5） ==========
        task_logger.log("\n🎨 阶段3: 批量生成海报图片（并发数: 5）")

        output_dir = storage.get_output_dir(user_id, product_id)
        poster_gen = PosterGenerator()

        results = []
        total = len(prompts)
        completed = 0

        # 并发生成单个海报的任务
        async def generate_single_poster(idx: int, poster_name: str, prompt_text: str):
            nonlocal completed
            task_logger.log(f"\n   [{idx + 1}/{total}] 开始生成: {poster_name}")

            try:
                result = await poster_gen.generate(
                    text_input=prompt_text,
                    image_paths=image_paths,
                    output_dir=str(output_dir),
                    filename_prefix=poster_name
                )

                if result.get("success") and result.get("saved_files"):
                    task_logger.log(f"      ✓ 成功: {poster_name} -> {result['saved_files']}")
                    return {
                        "name": poster_name,
                        "success": True,
                        "files": result["saved_files"]
                    }
                else:
                    task_logger.log(f"      ✗ 失败: {poster_name} - {result.get('error', '未知错误')}")
                    return {
                        "name": poster_name,
                        "success": False,
                        "error": result.get("error")
                    }

            except Exception as e:
                task_logger.log(f"      ✗ 异常: {poster_name} - {str(e)}")
                return {
                    "name": poster_name,
                    "success": False,
                    "error": str(e)
                }

        # 使用信号量控制并发数为5
        semaphore = asyncio.Semaphore(5)

        async def generate_with_semaphore(idx: int, name: str, prompt: str):
            nonlocal completed
            async with semaphore:
                result = await generate_single_poster(idx, name, prompt)
                completed += 1
                progress = 40 + int(completed / total * 50)
                update_task(
                    task_id,
                    progress=progress,
                    message=f"已完成 {completed}/{total} 张海报"
                )
                return result

        # 创建所有任务并并发执行
        tasks = [
            generate_with_semaphore(idx, name, prompt)
            for idx, (name, prompt) in enumerate(prompts.items())
        ]

        update_task(task_id, message=f"并发生成 {total} 张海报（5个同时）...")
        results = await asyncio.gather(*tasks)

        # ========== 阶段4: 完成 ==========
        success_count = sum(1 for r in results if r.get("success"))

        # 保存结果摘要
        summary = {
            "task_id": task_id,
            "user_id": user_id,
            "product_id": product_id,
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "results": results,
            "completed_at": datetime.now().isoformat()
        }
        summary_path = storage.save_summary(user_id, product_id, summary)

        task_logger.log("\n" + "=" * 60)
        task_logger.log("📊 生成统计")
        task_logger.log(f"   总数: {total}")
        task_logger.log(f"   成功: {success_count}")
        task_logger.log(f"   失败: {total - success_count}")
        task_logger.log(f"   摘要已保存: {summary_path}")
        task_logger.log("=" * 60)

        # 构建海报列表（供前端展示）
        posters = []
        for r in results:
            if r.get("success") and r.get("files"):
                for file_path in r["files"]:
                    posters.append({
                        "name": r["name"],
                        "file": os.path.basename(file_path),
                        "path": file_path
                    })

        update_task(
            task_id,
            status="completed",
            progress=100,
            message=f"完成！成功生成 {success_count}/{total} 张海报",
            posters=posters
        )

        task_logger.log("\n🎉 任务完成!")

    except Exception as e:
        error_msg = str(e)
        task_logger.log(f"\n❌ 任务失败: {error_msg}")

        import traceback
        task_logger.log(traceback.format_exc())

        update_task(
            task_id,
            status="failed",
            progress=0,
            message=f"生成失败: {error_msg}",
            error=error_msg
        )

    finally:
        task_logger.close()


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 API服务启动")
    yield
    logger.info("👋 API服务关闭")


app = FastAPI(
    title="AI电商海报生成系统",
    description="基于AI的电商海报批量生成API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ==================== API 路由 ====================

@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "message": "AI电商海报生成系统 API v1.0"}


@app.post("/api/upload-images")
async def upload_images(
    user_id: str = Form(...),
    product_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    上传产品图片
    """
    if len(files) > 9:
        raise HTTPException(status_code=400, detail="最多上传9张图片")

    # 创建用户目录
    upload_dir = UPLOAD_DIR / user_id / product_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for idx, file in enumerate(files):
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            continue

        # 保存文件
        ext = Path(file.filename).suffix or ".jpg"
        filename = f"{idx + 1}{ext}"
        file_path = upload_dir / filename

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        saved_files.append(str(file_path))
        logger.info(f"上传图片: {file_path}")

    return {
        "success": True,
        "user_id": user_id,
        "product_id": product_id,
        "uploaded_count": len(saved_files),
        "files": saved_files
    }


@app.post("/api/generate", response_model=TaskResponse)
async def start_generation(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    开始生成海报
    """
    # 检查是否有上传的图片
    upload_dir = UPLOAD_DIR / request.user_id / request.product_id
    if not upload_dir.exists():
        raise HTTPException(status_code=400, detail="请先上传产品图片")

    image_paths = list(upload_dir.glob("*"))
    if not image_paths:
        raise HTTPException(status_code=400, detail="请先上传产品图片")

    # 创建任务
    task = create_task(request.user_id, request.product_id)

    # 添加后台任务
    background_tasks.add_task(
        process_generation_task,
        task.task_id,
        request.user_id,
        request.product_id,
        request.product_description,
        [str(p) for p in image_paths]
    )

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        message="任务已创建，正在后台处理"
    )


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks_db[task_id]
    return task


@app.get("/api/tasks/{user_id}")
async def get_user_tasks(user_id: str):
    """获取用户的所有任务"""
    user_tasks = [
        task for task in tasks_db.values()
        if task.user_id == user_id
    ]
    return {"tasks": user_tasks}


@app.get("/api/poster/{user_id}/{product_id}/{filename}")
async def get_poster(user_id: str, product_id: str, filename: str):
    """获取生成的海报图片"""
    file_path = OUTPUT_DIR / user_id / product_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(file_path)


@app.get("/api/posters/{user_id}/{product_id}")
async def list_posters(user_id: str, product_id: str):
    """列出某个产品的所有海报"""
    output_dir = OUTPUT_DIR / user_id / product_id

    if not output_dir.exists():
        return {"posters": []}

    posters = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for file_path in output_dir.glob(ext):
            posters.append({
                "name": file_path.stem,
                "filename": file_path.name,
                "url": f"/api/poster/{user_id}/{product_id}/{file_path.name}"
            })

    return {"posters": posters}


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")

    del tasks_db[task_id]
    logger.info(f"删除任务: {task_id}")

    return {"success": True, "message": "任务已删除"}


# ==================== 编辑任务管理 ====================

def create_edit_task(request: EditRequest) -> EditTaskStatus:
    """创建新编辑任务"""
    import uuid
    task_id = "edit_" + str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    task = EditTaskStatus(
        task_id=task_id,
        user_id=request.user_id,
        product_id=request.product_id,
        poster_id=request.poster_id,
        status="pending",
        edit_type=request.edit_type,
        progress=0,
        message="编辑任务已创建，等待处理",
        created_at=now,
        updated_at=now,
        prompt=request.prompt,
        has_reference=request.reference_image is not None,
        parent_version=request.parent_version
    )

    edit_tasks_db[task_id] = task
    logger.info(f"创建编辑任务: {task_id} | 用户: {request.user_id} | 海报: {request.poster_id}")

    return task


def update_edit_task(task_id: str, **kwargs):
    """更新编辑任务状态"""
    if task_id not in edit_tasks_db:
        return

    task = edit_tasks_db[task_id]
    for key, value in kwargs.items():
        if hasattr(task, key):
            setattr(task, key, value)

    task.updated_at = datetime.now().isoformat()
    logger.info(f"更新编辑任务: {task_id} | 状态: {task.status} | 进度: {task.progress}%")


async def process_edit_task(
    task_id: str,
    user_id: str,
    product_id: str,
    poster_id: str,
    edit_type: str,
    prompt: str,
    source_image: str,
    reference_image: Optional[str] = None,
    parent_version: Optional[int] = None
):
    """后台执行海报编辑任务"""

    task_logger = TaskLogger(task_id, LOGS_DIR)

    try:
        task_logger.log("=" * 60)
        task_logger.log(f"🎨 开始处理编辑任务: {task_id}")
        task_logger.log(f"   用户ID: {user_id}")
        task_logger.log(f"   海报ID: {poster_id}")
        task_logger.log(f"   编辑类型: {edit_type}")
        task_logger.log(f"   提示词: {prompt[:100]}...")
        task_logger.log("=" * 60)

        # 初始化版本历史（如果需要）
        versions = version_manager.get_versions(user_id, product_id, poster_id)
        if versions["total_versions"] == 0:
            # 从输出目录找到原始海报并初始化
            output_dir = OUTPUT_DIR / user_id / product_id
            if output_dir.exists():
                # 查找匹配的海报文件
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                    for file_path in output_dir.glob(ext):
                        if poster_id in file_path.name or file_path.stem.startswith(poster_id.replace("poster_", "海报")):
                            version_manager.initialize_from_poster(
                                user_id, product_id, poster_id,
                                str(file_path), file_path.stem
                            )
                            task_logger.log(f"   ✓ 初始化版本历史: {file_path.name}")
                            break
                    if versions["total_versions"] > 0:
                        break

        # 阶段1: 准备输入
        update_edit_task(task_id, status="processing", progress=10, message="准备输入文件...")
        task_logger.log("\n📁 阶段1: 准备输入文件")

        # 创建临时工作目录
        import tempfile
        import shutil
        work_dir = Path(tempfile.mkdtemp())

        try:
            # 保存源图片
            source_path = work_dir / "source.png"
            if source_image.startswith("data:"):
                source_path = PosterEditor.decode_base64_image(source_image, str(source_path))
            else:
                # 如果是URL，复制文件
                shutil.copy(source_image, source_path)

            task_logger.log(f"   ✓ 源图片已保存")

            # 保存参考图片
            reference_path = None
            if reference_image:
                if reference_image.startswith("data:"):
                    reference_path = PosterEditor.decode_base64_image(
                        reference_image,
                        str(work_dir / "reference.png")
                    )
                else:
                    reference_path = reference_image
                task_logger.log(f"   ✓ 参考图片已保存")

            # 阶段2: 生成图片
            update_edit_task(task_id, status="generating", progress=30, message="正在生成新图片...")
            task_logger.log("\n🎨 阶段2: 调用AI生成图片")

            output_dir = storage.get_output_dir(user_id, product_id)
            poster_editor = PosterEditor()

            result = await poster_editor.edit_poster(
                source_image=str(source_path),
                prompt=prompt,
                reference_image=str(reference_path) if reference_path else None,
                output_dir=str(output_dir),
                filename_prefix=f"{poster_id}_v{versions['total_versions'] + 1:02d}"
            )

            if not result.get("success"):
                raise Exception(result.get("error", "生成失败"))

            generated_path = result["saved_files"][0] if result.get("saved_files") else None
            if not generated_path:
                raise Exception("未生成任何图片")

            task_logger.log(f"   ✓ 图片已生成: {generated_path}")

            # 阶段3: 保存版本
            update_edit_task(task_id, progress=80, message="保存版本...")
            task_logger.log("\n💾 阶段3: 保存版本历史")

            new_version = version_manager.create_version(
                user_id=user_id,
                product_id=product_id,
                poster_id=poster_id,
                version_type="full_edit" if edit_type == "full" else "partial_edit",
                original_image=str(source_path),
                generated_image=generated_path,
                prompt=prompt,
                reference_image=str(reference_path) if reference_path else None,
                parent_version=parent_version
            )

            task_logger.log(f"   ✓ 版本已保存: v{new_version:02d}")

            # 阶段4: 完成
            result_url = f"/api/edit/version/{user_id}/{product_id}/{poster_id}/v{new_version:02d}/full"

            update_edit_task(
                task_id,
                status="completed",
                progress=100,
                message=f"编辑完成，已创建版本 v{new_version:02d}",
                result_version=new_version,
                result_image_url=result_url
            )

            task_logger.log("\n🎉 编辑任务完成!")
            task_logger.log(f"   新版本: v{new_version:02d}")

        finally:
            # 清理临时目录
            shutil.rmtree(work_dir, ignore_errors=True)

    except Exception as e:
        error_msg = str(e)
        task_logger.log(f"\n❌ 编辑任务失败: {error_msg}")

        import traceback
        task_logger.log(traceback.format_exc())

        update_edit_task(
            task_id,
            status="failed",
            progress=0,
            message=f"编辑失败: {error_msg}",
            error=error_msg
        )

    finally:
        task_logger.close()


# ==================== 编辑API路由 ====================

@app.post("/api/edit/poster", response_model=EditTaskResponse)
async def edit_poster(
    request: EditRequest,
    background_tasks: BackgroundTasks
):
    """
    提交海报编辑请求（全图修改或局部修改）

    请求参数:
    - user_id: 用户ID
    - product_id: 产品ID
    - poster_id: 海报ID
    - edit_type: 编辑类型 ("full" 或 "partial")
    - prompt: 修改提示词
    - source_image: 源图片 (base64 data URI 或 URL)
    - reference_image: 参考图片 (可选，base64 data URI 或 URL)
    - parent_version: 基于哪个版本编辑 (可选，默认最新版本)
    """
    # 验证编辑类型
    if request.edit_type not in ["full", "partial"]:
        raise HTTPException(status_code=400, detail="edit_type 必须是 'full' 或 'partial'")

    # 创建编辑任务
    task = create_edit_task(request)

    # 添加后台任务
    background_tasks.add_task(
        process_edit_task,
        task.task_id,
        request.user_id,
        request.product_id,
        request.poster_id,
        request.edit_type,
        request.prompt,
        request.source_image,
        request.reference_image,
        request.parent_version
    )

    return EditTaskResponse(
        success=True,
        task_id=task.task_id,
        message="编辑任务已创建，正在后台处理",
        estimated_time_seconds=30
    )


@app.get("/api/edit/task/{task_id}")
async def get_edit_task_status(task_id: str):
    """获取编辑任务状态"""
    if task_id not in edit_tasks_db:
        raise HTTPException(status_code=404, detail="编辑任务不存在")

    return edit_tasks_db[task_id]


@app.get("/api/edit/versions/{user_id}/{product_id}/{poster_id}")
async def get_edit_versions(user_id: str, product_id: str, poster_id: str):
    """获取海报的所有版本"""
    base_url_prefix = f"/api/edit/version/{user_id}/{product_id}/{poster_id}"
    versions = version_manager.get_versions(user_id, product_id, poster_id, base_url_prefix)

    # 构建响应，移除内部字段
    response = {
        "poster_id": versions.get("poster_id"),
        "current_version": versions.get("current_version"),
        "total_versions": versions.get("total_versions"),
        "versions": versions.get("versions", [])
    }

    return response


@app.get("/api/edit/version/{user_id}/{product_id}/{poster_id}/{version}/{size}")
async def get_edit_version_image(
    user_id: str,
    product_id: str,
    poster_id: str,
    version: str,
    size: str = "full"
):
    """
    获取指定版本的图片

    参数:
    - version: 版本号 (如 "v01", "v02") 或纯数字
    - size: 图片大小 ("full" 或 "thumbnail")
    """
    # 处理版本号格式
    if version.startswith("v"):
        version_num = int(version[1:])
    else:
        version_num = int(version)

    # 验证size参数
    if size not in ["full", "thumbnail"]:
        size = "full"

    image_path = version_manager.get_version_image(
        user_id, product_id, poster_id, version_num, size
    )

    if not image_path:
        raise HTTPException(status_code=404, detail="版本图片不存在")

    return FileResponse(image_path)


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
