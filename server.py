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
import random
import string
import re
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Header
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
from backend.utils.auth import create_access_token, get_current_user
from backend.utils.verification import save_code, verify_code
from backend.utils.email_sender import send_verification_email
from backend.models.edit_models import EditRequest, EditTaskStatus, EditTaskResponse
from backend.db import init_db, crud

# ==================== 配置 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
VERSIONS_DIR = DATA_DIR / "versions"
FRONTEND_DIR = BASE_DIR / "frontend"

# 确保目录存在
for dir_path in [DATA_DIR, LOGS_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 设置日志
logger = setup_logger("api", LOGS_DIR / "api.log")

# API配置
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://ent2.zetatechs.com/v1")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123456")


# ==================== 辅助函数 ====================

def generate_session_id() -> str:
    """生成 session_{YYYYMMDDHHMMSS}_{4位随机数}"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"session_{timestamp}_{random_suffix}"


def get_poster_index(poster_name: str) -> int:
    """从海报名称提取序号（海报01 -> 0）"""
    match = re.search(r'海报\s*(\d+)', poster_name)
    if match:
        return int(match.group(1)) - 1  # 转为0开始
    return -1


async def generate_product_name(product_description: str) -> str:
    """使用小模型生成简洁的产品名称"""
    try:
        system_prompt = "你是一个产品命名专家。请从产品描述中提取或生成一个简洁的产品名称，2-6个中文字符。只返回产品名称，不要其他内容。"
        user_prompt = f"产品描述：\n{product_description}\n\n请生成产品名称："

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gemini-3-flash-preview-nothinking",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 20
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"生成产品名称失败: {e}，使用默认名称")
        return "未命名产品"


def save_product_name(session_dir: Path, name: str):
    """保存产品名称到文件"""
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "productname.txt").write_text(name, encoding='utf-8')


# ==================== 数据模型 ====================

class TaskStatus(BaseModel):
    task_id: str
    user_id: str
    session_id: str  # 改为 session_id
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
    product_description: str
    # product_id 移除，后端生成 session_id


class TaskResponse(BaseModel):
    success: bool
    task_id: Optional[str] = None
    session_id: Optional[str] = None  # 新增 session_id
    message: str


class SendCodeRequest(BaseModel):
    email: str


class LoginRequest(BaseModel):
    email: str
    code: str


# ==================== 任务管理 ====================

# 内存中的任务状态（生产环境应使用Redis或数据库）
tasks_db: dict[str, TaskStatus] = {}
edit_tasks_db: dict[str, EditTaskStatus] = {}
version_manager = VersionManager(DATA_DIR)
storage = StorageManager(DATA_DIR)


def create_task(user_id: str, session_id: str) -> TaskStatus:
    """创建新任务"""
    task_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    task = TaskStatus(
        task_id=task_id,
        user_id=user_id,
        session_id=session_id,
        status="pending",
        progress=0,
        message="任务已创建，等待处理",
        created_at=now,
        updated_at=now
    )

    tasks_db[task_id] = task
    logger.info(f"创建任务: {task_id} | 用户: {user_id} | 会话: {session_id}")

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
    session_id: str,
    product_description: str,
    image_paths: List[str],
    selected_posters: List[int] = None
):
    """后台执行完整的海报生成流程

    Args:
        selected_posters: 用户选择的海报序号列表（0-9），None表示生成全部
    """

    # 创建任务专属日志器
    task_logger = TaskLogger(task_id, LOGS_DIR)

    # 积分相关变量
    required_credits = 0
    credits_frozen = False
    generation_started = False  # 标记海报生成是否已开始（开始后积分逐个处理，异常时不再整体退回）

    try:
        task_logger.log("=" * 60)
        task_logger.log(f"🚀 开始处理任务: {task_id}")
        task_logger.log(f"   用户ID: {user_id}")
        task_logger.log(f"   会话ID: {session_id}")
        task_logger.log(f"   产品描述: {product_description[:100]}...")
        task_logger.log(f"   图片数量: {len(image_paths)}")
        task_logger.log("=" * 60)

        # ========== 积分冻结 ==========
        poster_count = len(selected_posters) if selected_posters else 10
        required_credits = poster_count * 4
        task_logger.log(f"\n💰 积分处理: 需要冻结 {required_credits} 积分（{poster_count}张 × 4积分）")

        if not crud.freeze_credits(int(user_id), required_credits):
            task_logger.log(f"   ❌ 积分冻结失败：积分不足")
            update_task(task_id, status="failed", progress=0, message="积分不足", error="积分不足")
            return

        credits_frozen = True
        task_logger.log(f"   ✓ 积分已冻结: {required_credits}")
        # ========== 积分冻结结束 ==========

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
        design_path = storage.save_design_output(user_id, session_id, design_output)
        task_logger.log(f"   ✓ 设计输出已保存: {design_path}")
        task_logger.log(f"   ✓ 输出长度: {len(design_output)} 字符")

        # 生成并保存产品名称
        task_logger.log("\n🏷️ 生成产品名称...")
        product_name = await generate_product_name(product_description)
        session_dir = OUTPUT_DIR / user_id / session_id
        save_product_name(session_dir, product_name)
        task_logger.log(f"   ✓ 产品名称: {product_name}")

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
        prompts_path = storage.save_prompts(user_id, session_id, prompts)
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

        output_dir = storage.get_output_dir(user_id, session_id)
        poster_gen = PosterGenerator()

        # 过滤：只生成用户选择的海报
        task_logger.log(f"\n   🎯 过滤海报（selected_posters={selected_posters}）")
        filtered_prompts = []
        for idx, (name, prompt) in enumerate(prompts.items()):
            poster_index = get_poster_index(name)
            if poster_index < 0:
                poster_index = idx

            # 如果用户指定了选择列表，只生成选中的；否则生成全部
            should_generate = selected_posters is None or poster_index in selected_posters
            task_logger.log(f"      {name}: poster_index={poster_index}, 是否生成={should_generate}")

            if should_generate:
                filtered_prompts.append((poster_index, name, prompt))

        task_logger.log(f"   📋 用户选择生成: {len(filtered_prompts)}/{len(prompts)} 张海报")
        if selected_posters is not None:
            task_logger.log(f"      选择序号: {sorted(selected_posters)}")

        results = []
        total = len(filtered_prompts)
        completed = 0

        # 并发生成单个海报的任务
        async def generate_single_poster(task_idx: int, poster_index: int, poster_name: str, prompt_text: str):
            nonlocal completed
            task_logger.log(f"\n   [{task_idx + 1}/{total}] 开始生成: {poster_name}")

            try:
                # 使用新的文件命名格式：XX-prime.jpg
                filename_prefix = f"{poster_index:02d}-prime"

                result = await poster_gen.generate(
                    text_input=prompt_text,
                    image_paths=image_paths,
                    output_dir=str(output_dir),
                    filename_prefix=filename_prefix
                )

                if result.get("success") and result.get("saved_files"):
                    task_logger.log(f"      ✓ 成功: {poster_name} -> {result['saved_files']}")
                    # 扣除冻结的4积分
                    crud.deduct_frozen(int(user_id), 4)
                    task_logger.log(f"      💰 已扣除4积分")
                    return {
                        "name": poster_name,
                        "success": True,
                        "files": result["saved_files"]
                    }
                else:
                    task_logger.log(f"      ✗ 失败: {poster_name} - {result.get('error', '未知错误')}")
                    # 退回冻结的4积分
                    crud.unfreeze_credits(int(user_id), 4)
                    task_logger.log(f"      💰 已退回4积分")
                    return {
                        "name": poster_name,
                        "success": False,
                        "error": result.get("error")
                    }

            except Exception as e:
                task_logger.log(f"      ✗ 异常: {poster_name} - {str(e)}")
                # 异常时退回冻结的4积分
                crud.unfreeze_credits(int(user_id), 4)
                task_logger.log(f"      💰 已退回4积分")
                return {
                    "name": poster_name,
                    "success": False,
                    "error": str(e)
                }

        # 使用信号量控制并发数为5
        semaphore = asyncio.Semaphore(5)

        async def generate_with_semaphore(task_idx: int, poster_index: int, name: str, prompt: str):
            nonlocal completed
            async with semaphore:
                result = await generate_single_poster(task_idx, poster_index, name, prompt)
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
            generate_with_semaphore(task_idx, poster_index, name, prompt)
            for task_idx, (poster_index, name, prompt) in enumerate(filtered_prompts)
        ]

        update_task(task_id, message=f"并发生成 {total} 张海报（5个同时）...")
        generation_started = True  # 标记开始生成，后续积分逐个处理
        results = await asyncio.gather(*tasks)

        # ========== 阶段4: 完成 ==========
        success_count = sum(1 for r in results if r.get("success"))

        # 保存结果摘要
        summary = {
            "task_id": task_id,
            "user_id": user_id,
            "session_id": session_id,
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "results": results,
            "completed_at": datetime.now().isoformat()
        }
        summary_path = storage.save_summary(user_id, session_id, summary)

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

        # 如果积分已冻结但任务失败（且海报生成尚未开始），退回所有冻结积分
        # 注：如果已开始生成，每个海报的积分已在内部处理，无需再退回
        if credits_frozen and required_credits > 0 and not generation_started:
            crud.unfreeze_credits(int(user_id), required_credits)
            task_logger.log(f"💰 任务异常，已退回全部冻结积分: {required_credits}")

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
    # 初始化数据库
    init_db()
    logger.info("📦 数据库初始化完成")
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


@app.get("/login")
async def login_page():
    """返回登录页面"""
    return FileResponse(FRONTEND_DIR / "login.html")


@app.get("/admin-codes")
async def admin_codes_page():
    """返回兑换码管理页面"""
    return FileResponse(FRONTEND_DIR / "admin-codes.html")


@app.get("/credits")
async def credits_page():
    """返回积分管理页面"""
    return FileResponse(FRONTEND_DIR / "credits.html")


@app.get("/edit-test")
async def edit_test_page():
    """返回图片编辑测试页面"""
    return FileResponse(FRONTEND_DIR / "edit-test.html")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "message": "AI电商海报生成系统 API v1.0"}


# ==================== 认证路由 ====================

@app.post("/api/auth/send-code")
async def send_verification_code(request: SendCodeRequest):
    """发送验证码"""
    code, success, remaining = save_code(request.email)

    if not success:
        raise HTTPException(
            status_code=429,
            detail=f"请等待{remaining}秒后再试"
        )

    # 发送邮件
    email_sent = send_verification_email(request.email, code)
    logger.info(f"📧 验证码: {code} | 邮箱: {request.email} | 发送状态: {'成功' if email_sent else '失败'}")

    return {
        "success": True,
        "message": "验证码已发送到邮箱（有效期1分钟）"
    }


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """登录验证"""
    # 验证验证码
    if not verify_code(request.email, request.code):
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    # 获取或创建用户（自动注册）
    user = crud.get_or_create_user(request.email)

    # 生成 JWT token
    access_token = create_access_token(user.id, user.email)

    logger.info(f"✅ 用户登录成功 | ID: {user.id} | 邮箱: {user.email}")

    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "credits": user.credits,
            "frozen": user.frozen
        }
    }


@app.get("/api/auth/me")
async def get_me(current_user = Depends(get_current_user)):
    """获取当前用户信息"""
    return current_user


# ==================== 兑换码 API ====================

def generate_redemption_code() -> str:
    """生成16位兑换码（大写字母+数字）"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=16))


class GenerateCodesRequest(BaseModel):
    """生成兑换码请求"""
    credits: int  # 积分数量（100/500/1000）
    count: int    # 生成数量（1-50）


class RedeemRequest(BaseModel):
    """兑换请求"""
    code: str


@app.post("/api/admin/generate-codes")
async def admin_generate_codes(
    request: GenerateCodesRequest,
    admin_password: str = Form(None)
):
    """
    管理员批量生成兑换码
    需要在请求头或表单中提供 Admin-Password
    """
    # 验证积分类型
    if request.credits not in [100, 500, 1000]:
        raise HTTPException(status_code=400, detail="积分类型必须是 100、500 或 1000")

    # 验证数量
    if not 1 <= request.count <= 50:
        raise HTTPException(status_code=400, detail="生成数量必须在 1-50 之间")

    # 生成兑换码
    codes = []
    for _ in range(request.count):
        code = generate_redemption_code()
        # 确保唯一性
        while crud.get_code_by_code(code):
            code = generate_redemption_code()
        crud.create_redemption_code(code, request.credits)
        codes.append(code)

    logger.info(f"🎫 生成 {request.count} 个 {request.credits} 积分兑换码")

    return {"success": True, "codes": codes, "credits": request.credits}


@app.get("/api/admin/codes")
async def admin_list_codes(
    admin_password: str = Header(None, alias="Admin-Password"),
    limit: int = 100
):
    """
    管理员获取兑换码列表
    需要在请求头中提供 Admin-Password
    """
    if admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="管理员密码错误")

    codes = crud.get_all_codes(limit)
    return {"success": True, "codes": codes}


@app.post("/api/credits/redeem")
async def redeem_credits(
    request: RedeemRequest,
    current_user = Depends(get_current_user)
):
    """
    用户兑换积分
    需要JWT认证
    """
    code = request.code.strip().upper()

    if not code:
        raise HTTPException(status_code=400, detail="请输入兑换码")

    result = crud.redeem_code(code, current_user.id)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    logger.info(f"🎁 用户 {current_user.email} 兑换 {result['credits']} 积分 | 兑换码: {code}")

    return result


@app.post("/api/admin/generate-codes-auth")
async def admin_generate_codes_with_header(
    request: GenerateCodesRequest,
    admin_password: str = Header(None, alias="Admin-Password")
):
    """
    管理员批量生成兑换码（通过请求头验证）
    """
    if admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="管理员密码错误")

    # 验证积分类型
    if request.credits not in [100, 500, 1000]:
        raise HTTPException(status_code=400, detail="积分类型必须是 100、500 或 1000")

    # 验证数量
    if not 1 <= request.count <= 50:
        raise HTTPException(status_code=400, detail="生成数量必须在 1-50 之间")

    # 生成兑换码
    codes = []
    for _ in range(request.count):
        code = generate_redemption_code()
        while crud.get_code_by_code(code):
            code = generate_redemption_code()
        crud.create_redemption_code(code, request.credits)
        codes.append(code)

    logger.info(f"🎫 生成 {request.count} 个 {request.credits} 积分兑换码")

    return {"success": True, "codes": codes, "credits": request.credits}


@app.post("/api/upload-images")
async def upload_images(
    user_id: str = Form(...),
    upload_id: str = Form(...),  # 临时上传ID（前端生成的时间戳）
    files: List[UploadFile] = File(...)
):
    """
    上传产品图片
    """
    if len(files) > 9:
        raise HTTPException(status_code=400, detail="最多上传9张图片")

    # 创建用户上传目录
    upload_dir = UPLOAD_DIR / user_id / upload_id
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
        "upload_id": upload_id,
        "uploaded_count": len(saved_files),
        "files": saved_files
    }


class GenerateRequestWithUploadId(BaseModel):
    user_id: str
    upload_id: str  # 临时上传ID
    product_description: str
    selected_posters: List[int] = None  # 用户选择的海报序号列表，None表示全选


@app.post("/api/generate", response_model=TaskResponse)
async def start_generation(
    request: GenerateRequestWithUploadId,
    background_tasks: BackgroundTasks
):
    """
    开始生成海报
    """
    # 检查是否有上传的图片
    upload_dir = UPLOAD_DIR / request.user_id / request.upload_id
    if not upload_dir.exists():
        raise HTTPException(status_code=400, detail="请先上传产品图片")

    image_paths = list(upload_dir.glob("*"))
    if not image_paths:
        raise HTTPException(status_code=400, detail="请先上传产品图片")

    # ========== 积分前置检查 ==========
    # 计算所需积分（4积分/张）
    poster_count = len(request.selected_posters) if request.selected_posters else 10
    required_credits = poster_count * 4

    # 获取用户积分
    user_credits = crud.get_user_credits(int(request.user_id))
    if user_credits.credits < required_credits:
        raise HTTPException(
            status_code=400,
            detail=f"积分不足，需要{required_credits}积分，当前可用{user_credits.credits}积分"
        )
    # ========== 积分检查结束 ==========

    # 生成 session_id
    session_id = generate_session_id()
    logger.info(f"生成 session_id: {session_id}")
    logger.info(f"收到 selected_posters: {request.selected_posters} (类型: {type(request.selected_posters)})")

    # 创建任务
    task = create_task(request.user_id, session_id)

    # 添加后台任务
    background_tasks.add_task(
        process_generation_task,
        task.task_id,
        request.user_id,
        session_id,
        request.product_description,
        [str(p) for p in image_paths],
        request.selected_posters  # 传递用户选择的海报列表
    )

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        session_id=session_id,  # 返回 session_id
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


@app.get("/api/poster/{user_id}/{session_id}/{filename}")
async def get_poster(user_id: str, session_id: str, filename: str):
    """获取生成的海报图片"""
    file_path = OUTPUT_DIR / user_id / session_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(file_path)


@app.get("/api/posters/{user_id}/{session_id}")
async def list_posters(user_id: str, session_id: str):
    """列出某个会话的所有海报"""
    output_dir = OUTPUT_DIR / user_id / session_id

    if not output_dir.exists():
        return {"product_name": None, "posters": []}

    # 读取产品名称
    productname_file = output_dir / "productname.txt"
    product_name = None
    if productname_file.exists():
        product_name = productname_file.read_text(encoding='utf-8').strip()

    posters = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for file_path in output_dir.glob(ext):
            # 只匹配 XX-prime.jpg 格式的文件
            match = re.match(r'^(\d+)-prime\.(jpg|jpeg|png|webp)$', file_path.name, re.IGNORECASE)
            if match:
                poster_index = int(match.group(1))
                posters.append({
                    "index": poster_index,
                    "filename": file_path.name,
                    "url": f"/api/poster/{user_id}/{session_id}/{file_path.name}"
                })

    # 按序号排序
    posters.sort(key=lambda x: x["index"])

    return {"product_name": product_name, "posters": posters}


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")

    del tasks_db[task_id]
    logger.info(f"删除任务: {task_id}")

    return {"success": True, "message": "任务已删除"}


# ==================== 历史记录 API ====================

@app.get("/api/history/{user_id}")
async def list_user_history(user_id: str, page: int = 0, page_size: int = 20):
    """
    获取用户的历史记录列表（分页）

    参数:
    - user_id: 用户ID
    - page: 页码（从0开始）
    - page_size: 每页条数（默认20）
    """
    user_dir = OUTPUT_DIR / user_id
    if not user_dir.exists():
        return {"total": 0, "page": page, "page_size": page_size, "has_more": False, "sessions": []}

    sessions = []
    for session_dir in user_dir.iterdir():
        if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
            continue

        session_id = session_dir.name

        # 解析时间戳 (session_YYYYMMDDHHMMSS_xxxx)
        try:
            timestamp_str = session_id.split("_")[1]
            created_at = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except:
            created_at = datetime.now()

        # 读取产品名称
        productname_file = session_dir / "productname.txt"
        product_name = "未命名产品"
        if productname_file.exists():
            product_name = productname_file.read_text(encoding='utf-8').strip()

        # 查找封面图 (00-prime.*)
        cover_image = None
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            cover_file = session_dir / f"00-prime{ext}"
            if cover_file.exists():
                cover_image = f"/api/poster/{user_id}/{session_id}/00-prime{ext}"
                break

        # 统计海报数量
        poster_count = len(list(session_dir.glob("*-prime.*")))

        sessions.append({
            "session_id": session_id,
            "product_name": product_name,
            "cover_image": cover_image,
            "poster_count": poster_count,
            "created_at": created_at.isoformat()
        })

    # 按时间倒序排序
    sessions.sort(key=lambda x: x["created_at"], reverse=True)

    # 分页
    total = len(sessions)
    start = page * page_size
    end = start + page_size
    paged_sessions = sessions[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": end < total,
        "sessions": paged_sessions
    }


@app.get("/api/history/{user_id}/{session_id}")
async def get_history_detail(user_id: str, session_id: str):
    """
    获取历史记录详情（与 /api/posters 格式兼容）
    """
    session_dir = OUTPUT_DIR / user_id / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="历史记录不存在")

    # 读取产品名称
    productname_file = session_dir / "productname.txt"
    product_name = "未命名产品"
    if productname_file.exists():
        product_name = productname_file.read_text(encoding='utf-8').strip()

    # 解析创建时间
    try:
        timestamp_str = session_id.split("_")[1]
        created_at = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S").isoformat()
    except:
        created_at = datetime.now().isoformat()

    # 查找所有海报
    posters = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for file_path in session_dir.glob(ext):
            match = re.match(r'^(\d+)-prime\.(jpg|jpeg|png|webp)$', file_path.name, re.IGNORECASE)
            if match:
                poster_index = int(match.group(1))
                poster_id = f"{poster_index:02d}"

                # 查询版本数量
                version_count = 1  # 至少有原始版本
                version_file = VERSIONS_DIR / user_id / session_id / f"poster_{poster_id}_versions.json"
                if version_file.exists():
                    try:
                        with open(version_file, 'r', encoding='utf-8') as f:
                            version_data = json.load(f)
                            version_count = len(version_data.get('versions', []))
                    except:
                        pass

                posters.append({
                    "index": poster_index,
                    "filename": file_path.name,
                    "url": f"/api/poster/{user_id}/{session_id}/{file_path.name}",
                    "version_count": version_count
                })

    # 按序号排序
    posters.sort(key=lambda x: x["index"])

    return {
        "session_id": session_id,
        "product_name": product_name,
        "created_at": created_at,
        "posters": posters
    }


@app.get("/history")
async def history_page():
    """返回历史记录页面"""
    return FileResponse(FRONTEND_DIR / "history.html")


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


def resolve_image_url_to_path(image_url: str) -> Optional[str]:
    """
    将 API URL 转换为实际文件路径
    支持: /api/poster/..., /api/edit/version/...
    """
    if image_url.startswith("/api/poster/"):
        # 格式: /api/poster/{user_id}/{session_id}/{filename}
        parts = image_url.split("/")
        if len(parts) >= 5:
            user_id = parts[3]
            session_id = parts[4]
            filename = parts[5]
            file_path = OUTPUT_DIR / user_id / session_id / filename
            if file_path.exists():
                return str(file_path)
    elif image_url.startswith("/api/edit/version/"):
        # 格式: /api/edit/version/{user_id}/{product_id}/{poster_id}/v{version}/{size}
        parts = image_url.split("/")
        if len(parts) >= 8:
            url_user_id = parts[4]
            url_product_id = parts[5]
            url_poster_id = parts[6]
            url_version = int(parts[7].replace("v", ""))
            version_image_path = version_manager.get_version_image(
                url_user_id, url_product_id, url_poster_id, url_version, "full"
            )
            return version_image_path
    return None


async def process_edit_task(
    task_id: str,
    user_id: str,
    product_id: str,
    poster_id: str,
    edit_type: str,
    prompt: str,
    source_image: str,
    original_image: Optional[str] = None,  # 局部修改时的原图（无标记）
    reference_image: Optional[str] = None,
    parent_version: Optional[int] = None
):
    """后台执行海报编辑任务"""

    task_logger = TaskLogger(task_id, LOGS_DIR)

    # 积分相关变量
    credits_frozen = False

    try:
        task_logger.log("=" * 60)
        task_logger.log(f"🎨 开始处理编辑任务: {task_id}")
        task_logger.log(f"   用户ID: {user_id}")
        task_logger.log(f"   海报ID: {poster_id}")
        task_logger.log(f"   编辑类型: {edit_type}")
        task_logger.log(f"   提示词: {prompt[:100]}...")
        task_logger.log("=" * 60)

        # ========== 积分冻结 ==========
        task_logger.log(f"\n💰 积分处理: 需要冻结 4 积分")
        if not crud.freeze_credits(int(user_id), 4):
            task_logger.log(f"   ❌ 积分冻结失败：积分不足")
            update_edit_task(task_id, status="failed", progress=0, message="积分不足", error="积分不足")
            return
        credits_frozen = True
        task_logger.log(f"   ✓ 积分已冻结: 4")
        # ========== 积分冻结结束 ==========

        # 初始化版本历史（如果需要）
        versions = version_manager.get_versions(user_id, product_id, poster_id)
        if versions["total_versions"] == 0:
            # 从输出目录找到原始海报并初始化
            output_dir = OUTPUT_DIR / user_id / product_id
            if output_dir.exists():
                # 查找匹配的海报文件（格式：XX-prime.jpg）
                original_file = output_dir / f"{poster_id}-prime.jpg"
                if not original_file.exists():
                    # 尝试其他扩展名
                    for ext in [".jpeg", ".png", ".webp"]:
                        alt_file = output_dir / f"{poster_id}-prime{ext}"
                        if alt_file.exists():
                            original_file = alt_file
                            break

                if original_file.exists():
                    version_manager.initialize_from_poster(
                        user_id, product_id, poster_id,
                        str(original_file), f"海报{int(poster_id)+1:02d}"
                    )
                    task_logger.log(f"   ✓ 初始化版本历史: {original_file.name}")
                else:
                    task_logger.log(f"   ⚠️ 未找到原始海报: {poster_id}-prime.jpg")

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
            elif source_image.startswith("/api/"):
                # 解析 API URL 为实际文件路径
                resolved_path = resolve_image_url_to_path(source_image)
                if resolved_path:
                    shutil.copy(resolved_path, source_path)
                    task_logger.log(f"   ✓ 源图片已从 API URL 获取")
                else:
                    raise Exception(f"无法解析源图片 URL: {source_image}")
            else:
                # 如果是文件路径，复制文件
                shutil.copy(source_image, source_path)

            task_logger.log(f"   ✓ 源图片已保存")

            # 保存原图（局部修改时）
            original_path = None
            if original_image and edit_type == "partial":
                original_path = work_dir / "original.png"
                if original_image.startswith("data:"):
                    original_path = PosterEditor.decode_base64_image(
                        original_image,
                        str(original_path)
                    )
                elif original_image.startswith("/api/"):
                    # 解析 API URL 为实际文件路径
                    resolved_path = resolve_image_url_to_path(original_image)
                    if resolved_path:
                        shutil.copy(resolved_path, original_path)
                        original_path = str(original_path)
                        task_logger.log(f"   ✓ 原图已从 API URL 获取")
                    else:
                        task_logger.log(f"   ⚠️ 无法解析原图 URL: {original_image}")
                        original_path = None
                else:
                    # 如果是文件路径，复制文件
                    shutil.copy(original_image, original_path)
                    original_path = str(original_path)
                if original_path:
                    task_logger.log(f"   ✓ 原图已保存")

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

            # 局部修改时，增强提示词
            final_prompt = prompt
            if edit_type == "partial":
                final_prompt = f"""请根据以下要求修改图片：

{prompt}

【重要说明】
- 第一张图是带有涂抹标记的图片，黄色/橙色半透明区域是需要修改的部分
- 第二张图是原始图片（无标记）
- 请只修改标记区域，保持其他区域不变
- 生成的图片不要保留涂抹标记，要生成干净的最终效果"""

            output_dir = storage.get_output_dir(user_id, product_id)
            poster_editor = PosterEditor()

            result = await poster_editor.edit_poster(
                source_image=str(source_path),
                original_image=str(original_path) if original_path else None,
                prompt=final_prompt,
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

            # ========== 积分扣除 ==========
            crud.deduct_frozen(int(user_id), 4)
            task_logger.log(f"   💰 已扣除4积分")

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

        # ========== 积分退回 ==========
        if credits_frozen:
            crud.unfreeze_credits(int(user_id), 4)
            task_logger.log(f"💰 编辑失败，已退回冻结积分: 4")

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

    # ========== 积分前置检查 ==========
    user_credits = crud.get_user_credits(int(request.user_id))
    if user_credits.credits < 4:
        raise HTTPException(
            status_code=400,
            detail=f"积分不足，编辑需要4积分，当前可用{user_credits.credits}积分"
        )
    # ========== 积分检查结束 ==========

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
        request.original_image,  # 局部修改时的原图
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
