#!/usr/bin/env python3
"""
编辑相关的数据模型
"""

from pydantic import BaseModel
from typing import Optional


class EditRequest(BaseModel):
    """编辑请求模型"""
    user_id: str
    product_id: str
    poster_id: str
    edit_type: str  # "full" 或 "partial"
    prompt: str
    source_image: str  # base64 data URI 或 URL（局部修改时为带标记的图）
    original_image: Optional[str] = None  # 局部修改时的原图（无标记）
    reference_image: Optional[str] = None  # base64 data URI 或 URL
    parent_version: Optional[int] = None


class EditTaskStatus(BaseModel):
    """编辑任务状态模型"""
    task_id: str
    user_id: str
    product_id: str
    poster_id: str
    status: str  # pending, processing, generating, completed, failed
    edit_type: str  # full, partial
    progress: int  # 0-100
    message: str
    created_at: str
    updated_at: str
    # 编辑参数
    prompt: Optional[str] = None
    has_reference: Optional[bool] = None
    parent_version: Optional[int] = None
    # 结果
    result_version: Optional[int] = None
    result_image_url: Optional[str] = None
    error: Optional[str] = None


class EditTaskResponse(BaseModel):
    """编辑任务响应模型"""
    success: bool
    task_id: Optional[str] = None
    message: str
    estimated_time_seconds: Optional[int] = None
