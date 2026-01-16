"""
数据库 CRUD 操作
"""
from typing import Optional, List
from datetime import datetime
from .database import get_db
from .models import User, UserCredits, Order


# ==================== 用户操作 ====================

def get_user_by_email(email: str) -> Optional[User]:
    """通过邮箱获取用户"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        if row:
            return User(**dict(row))
        return None


def get_user_by_id(user_id: int) -> Optional[User]:
    """通过ID获取用户"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if row:
            return User(**dict(row))
        return None


def create_user(email: str) -> User:
    """创建新用户"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (email) VALUES (?)',
            (email,)
        )
        user_id = cursor.lastrowid

    # 在新连接中获取用户（确保数据已提交）
    return get_user_by_id(user_id)


def get_or_create_user(email: str) -> User:
    """获取或创建用户"""
    user = get_user_by_email(email)
    if user:
        return user
    return create_user(email)


def get_user_credits(user_id: int) -> UserCredits:
    """获取用户积分"""
    user = get_user_by_id(user_id)
    if not user:
        raise ValueError(f"用户不存在: {user_id}")
    return UserCredits(
        credits=user.credits,
        frozen=user.frozen,
        total=user.credits + user.frozen
    )


def freeze_credits(user_id: int, amount: int) -> bool:
    """
    冻结积分（生成前调用）

    Args:
        user_id: 用户ID
        amount: 需要冻结的积分数量

    Returns:
        True 表示冻结成功，False 表示积分不足
    """
    with get_db() as conn:
        cursor = conn.cursor()
        # 检查可用积分是否足够
        cursor.execute('SELECT credits FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row or row['credits'] < amount:
            return False

        # 冻结积分
        cursor.execute('''
            UPDATE users
            SET credits = credits - ?, frozen = frozen + ?, updated_at = ?
            WHERE id = ?
        ''', (amount, amount, datetime.now().isoformat(), user_id))
        return True


def deduct_frozen(user_id: int, amount: int) -> bool:
    """
    扣除冻结积分（生成成功后调用）

    Args:
        user_id: 用户ID
        amount: 需要扣除的冻结积分数量

    Returns:
        True 表示扣除成功
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users
            SET frozen = frozen - ?, updated_at = ?
            WHERE id = ? AND frozen >= ?
        ''', (amount, datetime.now().isoformat(), user_id, amount))
        return cursor.rowcount > 0


def unfreeze_credits(user_id: int, amount: int) -> bool:
    """
    解冻积分（生成失败后调用）

    Args:
        user_id: 用户ID
        amount: 需要解冻的积分数量

    Returns:
        True 表示解冻成功
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users
            SET credits = credits + ?, frozen = frozen - ?, updated_at = ?
            WHERE id = ? AND frozen >= ?
        ''', (amount, amount, datetime.now().isoformat(), user_id, amount))
        return cursor.rowcount > 0


def add_credits(user_id: int, amount: int) -> bool:
    """
    增加积分（充值成功后调用）

    Args:
        user_id: 用户ID
        amount: 增加的积分数量

    Returns:
        True 表示增加成功
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users
            SET credits = credits + ?, updated_at = ?
            WHERE id = ?
        ''', (amount, datetime.now().isoformat(), user_id))
        return cursor.rowcount > 0


# ==================== 订单操作 ====================

def create_order(user_id: int, amount: int) -> Order:
    """
    创建充值订单

    Args:
        user_id: 用户ID
        amount: 充值积分数量

    Returns:
        创建的订单对象
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO orders (user_id, amount) VALUES (?, ?)',
            (user_id, amount)
        )
        order_id = cursor.lastrowid

    return get_order_by_id(order_id)


def get_order_by_id(order_id: int) -> Optional[Order]:
    """通过ID获取订单"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM orders WHERE id = ?', (order_id,))
        row = cursor.fetchone()
        if row:
            return Order(**dict(row))
        return None


def update_order_paid(order_id: int, trade_no: str) -> bool:
    """
    更新订单为已支付

    Args:
        order_id: 订单ID
        trade_no: 第三方支付单号

    Returns:
        True 表示更新成功
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE orders
            SET status = 'paid', trade_no = ?, updated_at = ?
            WHERE id = ? AND status = 'pending'
        ''', (trade_no, datetime.now().isoformat(), order_id))
        return cursor.rowcount > 0


def get_user_orders(user_id: int, limit: int = 20) -> List[Order]:
    """
    获取用户订单列表

    Args:
        user_id: 用户ID
        limit: 返回数量限制

    Returns:
        订单列表，按创建时间倒序
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?',
            (user_id, limit)
        )
        return [Order(**dict(row)) for row in cursor.fetchall()]


# ==================== 兑换码操作 ====================

def create_redemption_code(code: str, credits: int) -> dict:
    """
    创建兑换码

    Args:
        code: 兑换码字符串
        credits: 积分数量

    Returns:
        创建的兑换码记录
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO redemption_codes (code, credits) VALUES (?, ?)',
            (code, credits)
        )
        code_id = cursor.lastrowid

    return get_code_by_id(code_id)


def get_code_by_id(code_id: int) -> Optional[dict]:
    """通过ID获取兑换码"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM redemption_codes WHERE id = ?', (code_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_code_by_code(code: str) -> Optional[dict]:
    """通过兑换码字符串获取记录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM redemption_codes WHERE code = ?', (code,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def redeem_code(code: str, user_id: int) -> dict:
    """
    兑换积分

    Args:
        code: 兑换码
        user_id: 用户ID

    Returns:
        {"success": bool, "message": str, "credits": int}
    """
    code_record = get_code_by_code(code)

    if not code_record:
        return {"success": False, "message": "无效的兑换码", "credits": 0}

    if code_record["status"] == "used":
        return {"success": False, "message": "兑换码已被使用", "credits": 0}

    credits = code_record["credits"]

    with get_db() as conn:
        cursor = conn.cursor()
        # 更新兑换码状态
        cursor.execute('''
            UPDATE redemption_codes
            SET status = 'used', used_by = ?, used_at = ?
            WHERE code = ? AND status = 'unused'
        ''', (user_id, datetime.now().isoformat(), code))

        if cursor.rowcount == 0:
            return {"success": False, "message": "兑换失败，请重试", "credits": 0}

        # 增加用户积分
        cursor.execute('''
            UPDATE users
            SET credits = credits + ?, updated_at = ?
            WHERE id = ?
        ''', (credits, datetime.now().isoformat(), user_id))

    return {"success": True, "message": f"兑换成功，获得 {credits} 积分", "credits": credits}


def get_all_codes(limit: int = 100) -> List[dict]:
    """
    获取所有兑换码列表

    Args:
        limit: 返回数量限制

    Returns:
        兑换码列表，按创建时间倒序
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM redemption_codes ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
