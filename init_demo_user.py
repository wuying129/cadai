#!/usr/bin/env python3
"""初始化演示用户"""
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.db.database import init_db
from backend.db import crud
from backend.utils.auth import create_access_token

EMAIL = "zhanbingqiang@gpulink.cc"
TARGET_CREDITS = 10000


def main():
    print("=" * 50)
    print("初始化演示用户")
    print("=" * 50)

    # 初始化数据库
    init_db()

    # 获取或创建用户
    user = crud.get_or_create_user(EMAIL)
    print(f"\n用户已创建/获取:")
    print(f"  ID: {user.id}")
    print(f"  邮箱: {user.email}")
    print(f"  当前积分: {user.credits}")

    # 补足积分到目标值
    if user.credits < TARGET_CREDITS:
        add_amount = TARGET_CREDITS - user.credits
        crud.add_credits(user.id, add_amount)
        print(f"  补充积分: +{add_amount}")

    # 重新获取用户信息
    user = crud.get_user_by_id(user.id)

    # 生成 JWT token
    token = create_access_token(user.id, user.email)

    print(f"\n最终状态:")
    print(f"  积分: {user.credits}")
    print(f"  冻结: {user.frozen}")

    print("\n" + "=" * 50)
    print("请将以下内容填入 frontend/index.html")
    print("=" * 50)
    print(f"\nUSER_ID: {user.id}")
    print(f"EMAIL: {user.email}")
    print(f"CREDITS: {user.credits}")
    print(f"\nTOKEN:\n{token}")

    return user.id, token


if __name__ == "__main__":
    main()
