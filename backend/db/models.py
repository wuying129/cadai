"""
数据模型定义
"""
from pydantic import BaseModel, EmailStr
from typing import Optional


# ==================== 用户模型 ====================

class UserCreate(BaseModel):
    """创建用户请求"""
    email: EmailStr


class User(BaseModel):
    """用户完整信息"""
    id: int
    email: str
    credits: int
    frozen: int
    created_at: str
    updated_at: str


class UserCredits(BaseModel):
    """用户积分信息"""
    credits: int      # 可用积分
    frozen: int       # 冻结积分
    total: int        # 总积分 (credits + frozen)


# ==================== 订单模型 ====================

class OrderCreate(BaseModel):
    """创建订单请求"""
    user_id: int
    amount: int       # 充值积分


class Order(BaseModel):
    """订单完整信息"""
    id: int
    user_id: int
    amount: int
    status: str       # pending / paid
    trade_no: Optional[str] = None
    created_at: str
    updated_at: str
