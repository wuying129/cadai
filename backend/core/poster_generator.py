#!/usr/bin/env python3
"""
海报图片生成器
调用AI生成海报图片
"""

import os
import re
import base64
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import aiohttp
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==================== API 配置 ====================
BASE_URL = os.getenv("API_BASE_URL", "https://ent2.zetatechs.com/v1")
API_KEY = os.getenv("API_KEY", "")

# 图片生成模型配置
# 注：原首选模型 gemini-3-pro-image-preview-flatfee 不稳定，已改为备用
# 如需恢复，可在 .env 中设置 IMAGE_MODEL_NAME 和 FALLBACK_IMAGE_MODEL_NAME
MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gemini-3-pro-image-preview")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_IMAGE_MODEL_NAME", "gemini-3-pro-image-preview")
PRIMARY_TIMEOUT = 300  # 首选模型超时时间（秒）
FALLBACK_TIMEOUT = 45  # 备用模型超时时间（秒）


class PosterGenerator:
    """海报图片生成器"""

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = MODEL_NAME
        self.fallback_model_name = FALLBACK_MODEL_NAME
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def _get_media_type(self, image_path: str) -> str:
        """获取图片的媒体类型"""
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(suffix, "image/jpeg")
    
    def _build_message_content(self, text_input: str, image_paths: List[str] = None):
        """构建消息内容"""
        if not image_paths:
            return text_input
        
        # 过滤出存在的图片
        valid_images = [p for p in image_paths if os.path.exists(p)]
        
        if not valid_images:
            return text_input
        
        # 构建多模态内容
        content = []
        
        # 添加提示词
        content.append({
            "type": "text",
            "text": text_input
        })
        
        # 添加参考图片
        for img_path in valid_images:
            try:
                base64_data = self._encode_image_to_base64(img_path)
                media_type = self._get_media_type(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_data}"
                    }
                })
            except Exception as e:
                print(f"  ✗ 编码图片失败 {img_path}: {e}")
        
        return content
    
    def _extract_base64_from_response(self, content) -> List[dict]:
        """从响应中提取base64图片数据"""
        results = []
        
        if isinstance(content, str):
            # 匹配 Markdown 图片格式
            pattern = r'!\[.*?\]\((data:image/(\w+);base64,([A-Za-z0-9+/=]+))\)'
            matches = re.findall(pattern, content)
            
            for full_uri, img_type, base64_data in matches:
                results.append({
                    "type": img_type,
                    "data": base64_data
                })
            
            # 也尝试匹配纯 data URI
            if not results:
                pattern2 = r'data:image/(\w+);base64,([A-Za-z0-9+/=]+)'
                matches2 = re.findall(pattern2, content)
                for img_type, base64_data in matches2:
                    results.append({
                        "type": img_type,
                        "data": base64_data
                    })
        
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type", "")
                
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        results.extend(self._extract_base64_from_response(text))
                
                elif item_type == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        try:
                            header, base64_data = image_url.split(",", 1)
                            
                            img_type = "png"
                            if "jpeg" in header or "jpg" in header:
                                img_type = "jpeg"
                            elif "webp" in header:
                                img_type = "webp"
                            elif "gif" in header:
                                img_type = "gif"
                            
                            results.append({
                                "type": img_type,
                                "data": base64_data
                            })
                        except Exception:
                            pass
        
        return results
    
    def _save_images(self, images: List[dict], output_dir: str, filename_prefix: str) -> List[str]:
        """保存图片到文件"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 清理文件名
        safe_prefix = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', filename_prefix)
        
        for idx, img in enumerate(images):
            try:
                ext = img["type"]
                if ext == "jpeg":
                    ext = "jpg"
                
                if len(images) == 1:
                    filename = f"{safe_prefix}.{ext}"
                else:
                    filename = f"{safe_prefix}_{idx}.{ext}"
                
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(img["data"]))
                
                saved_files.append(filepath)
                print(f"  ✓ 图片已保存: {filepath}")
                
            except Exception as e:
                print(f"  ✗ 保存图片 {idx} 失败: {e}")
        
        return saved_files
    
    async def _call_api(
        self,
        session: aiohttp.ClientSession,
        model_name: str,
        message_content,
        timeout: int
    ) -> dict:
        """
        调用API生成图片

        Args:
            session: aiohttp会话
            model_name: 模型名称
            message_content: 消息内容
            timeout: 超时时间（秒）

        Returns:
            API响应数据
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "max_tokens": 4096
        }

        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API请求失败 ({response.status}): {error_text}")

            return await response.json()

    async def generate(
        self,
        text_input: str,
        image_paths: List[str] = None,
        output_dir: str = "./output",
        filename_prefix: str = "poster"
    ) -> dict:
        """
        生成海报图片（支持模型自动切换）

        首选模型60秒超时后自动切换到备用模型

        Args:
            text_input: 提示词文本
            image_paths: 参考图片路径列表
            output_dir: 输出目录
            filename_prefix: 文件名前缀

        Returns:
            包含结果的字典
        """
        print(f"\n🎨 开始生成: {filename_prefix}")

        # 检查图片路径
        valid_images = []
        if image_paths:
            for path in image_paths:
                if os.path.exists(path):
                    valid_images.append(path)

        # 构建消息内容
        message_content = self._build_message_content(text_input, valid_images if valid_images else None)

        result = {
            "success": False,
            "saved_files": [],
            "error": None,
            "raw_response": None,
            "model_used": None
        }

        # 模型列表：首选 -> 备用
        models_to_try = [
            (self.model_name, PRIMARY_TIMEOUT, "首选"),
            (self.fallback_model_name, FALLBACK_TIMEOUT, "备用")
        ]

        async with aiohttp.ClientSession() as session:
            for model_name, timeout, model_type in models_to_try:
                try:
                    print(f"  📡 使用{model_type}模型: {model_name} (超时: {timeout}秒)")

                    data = await self._call_api(session, model_name, message_content, timeout)
                    result["raw_response"] = data
                    result["model_used"] = model_name

                    # 提取生成的图片
                    if "choices" in data and len(data["choices"]) > 0:
                        message = data["choices"][0].get("message", {})
                        content = message.get("content", "")

                        # 提取图片
                        images = self._extract_base64_from_response(content)

                        if images:
                            saved_files = self._save_images(images, output_dir, filename_prefix)
                            result["saved_files"] = saved_files
                            result["success"] = len(saved_files) > 0

                            if saved_files:
                                print(f"  ✅ 成功生成 {len(saved_files)} 张图片 (模型: {model_name})")
                                return result
                            else:
                                print("  ⚠️ 未能保存图片")
                        else:
                            print("  ⚠️ 响应中未检测到图片数据")
                    else:
                        print("  ⚠️ 响应格式异常")

                    # 如果没有成功生成图片，继续尝试下一个模型

                except asyncio.TimeoutError:
                    print(f"  ⏱️ {model_type}模型超时 ({timeout}秒)，切换到下一个模型...")
                    continue
                except aiohttp.ClientError as e:
                    print(f"  ⚠️ {model_type}模型网络错误: {e}，切换到下一个模型...")
                    continue
                except Exception as e:
                    # 检查是否是429或5xx错误，这些情况应该切换模型
                    error_str = str(e)
                    if "429" in error_str or "503" in error_str or "502" in error_str or "负载" in error_str:
                        print(f"  ⚠️ {model_type}模型服务繁忙: {e}，切换到下一个模型...")
                        continue
                    else:
                        # 其他错误直接记录并继续
                        print(f"  ⚠️ {model_type}模型错误: {e}，切换到下一个模型...")
                        continue

        # 所有模型都失败了
        result["error"] = "所有模型均生成失败"
        print(f"  ❌ {result['error']}")
        return result
