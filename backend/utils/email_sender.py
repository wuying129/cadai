"""
SMTP 邮件发送
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# SMTP 配置（从环境变量读取）
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")


def send_verification_email(to_email: str, code: str) -> bool:
    """
    发送验证码邮件

    Args:
        to_email: 收件人邮箱
        code: 验证码

    Returns:
        是否发送成功
    """
    # 检查 SMTP 是否配置
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
        print(f"⚠️ SMTP未配置，验证码: {code} -> {to_email}")
        return False

    try:
        # 构建邮件
        msg = MIMEMultipart()
        msg['From'] = SMTP_FROM or SMTP_USER
        msg['To'] = to_email
        msg['Subject'] = '登录验证码'

        # 邮件正文
        body = f"""您好！

您的验证码是：{code}

验证码有效期为1分钟，请尽快使用。

如果这不是您的操作，请忽略此邮件。
"""

        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 发送邮件
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM or SMTP_USER, to_email, msg.as_string())

        print(f"✅ 邮件发送成功: {to_email}")
        return True

    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False
