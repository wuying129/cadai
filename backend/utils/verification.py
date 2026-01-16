"""
验证码管理（内存存储）
- 6位数字验证码
- 1分钟有效期
- 60秒内同一邮箱不能重复发送（防刷）
"""
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple

# 内存存储
verification_codes: dict[str, dict] = {}

# 配置
CODE_EXPIRE_MINUTES = 1  # 验证码有效期（分钟）
RESEND_COOLDOWN_SECONDS = 60  # 重新发送冷却时间（秒）


def generate_code() -> str:
    """生成6位数字验证码"""
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])


def save_code(email: str) -> Tuple[Optional[str], bool, int]:
    """
    保存验证码

    Args:
        email: 邮箱地址

    Returns:
        Tuple[code, success, remaining_seconds]
        - code: 验证码（如果成功）或 None（如果失败）
        - success: 是否成功
        - remaining_seconds: 剩余冷却秒数（如果在冷却中）
    """
    now = datetime.now()

    # 检查是否在60秒冷却期内
    if email in verification_codes:
        created = datetime.fromisoformat(verification_codes[email]["created_at"])
        elapsed = (now - created).total_seconds()

        if elapsed < RESEND_COOLDOWN_SECONDS:
            remaining = int(RESEND_COOLDOWN_SECONDS - elapsed)
            return None, False, remaining

    # 生成新验证码
    code = generate_code()
    verification_codes[email] = {
        "code": code,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=CODE_EXPIRE_MINUTES)).isoformat()
    }

    return code, True, 0


def verify_code(email: str, code: str) -> bool:
    """
    验证验证码

    Args:
        email: 邮箱地址
        code: 用户输入的验证码

    Returns:
        验证是否成功
    """
    if email not in verification_codes:
        return False

    stored = verification_codes[email]

    # 检查是否过期
    expires_at = datetime.fromisoformat(stored["expires_at"])
    if datetime.now() > expires_at:
        del verification_codes[email]
        return False

    # 验证码匹配
    if stored["code"] == code:
        del verification_codes[email]  # 验证成功后删除
        return True

    return False


def cleanup_expired():
    """清理过期验证码（可选：定时任务调用）"""
    now = datetime.now()
    expired = [
        email for email, data in verification_codes.items()
        if datetime.fromisoformat(data["expires_at"]) < now
    ]
    for email in expired:
        del verification_codes[email]
    return len(expired)
