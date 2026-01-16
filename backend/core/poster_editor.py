#!/usr/bin/env python3
"""
海报编辑器
处理海报的全图修改和局部修改
复用 PosterGenerator 的逻辑
"""

import os
import base64
from pathlib import Path
from typing import Optional

from backend.core.poster_generator import PosterGenerator


class PosterEditor:
    """海报编辑器 - 处理全图和局部修改"""

    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化编辑器

        Args:
            api_key: API密钥（可选，默认使用PosterGenerator的配置）
            base_url: API基础URL（可选）
        """
        self.generator = PosterGenerator(api_key=api_key, base_url=base_url)

    @staticmethod
    def decode_base64_image(base64_data: str, output_path: str) -> str:
        """
        解码并保存base64图片

        Args:
            base64_data: base64数据（可能包含data URI前缀）
            output_path: 输出文件路径

        Returns:
            保存的文件路径
        """
        # 处理 data URI 格式
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]

        # 检测图片格式
        if "data:image" in base64_data:
            # 提取格式
            format_part = base64_data.split(":")[1].split(";")[0] if ":" in base64_data else "png"
            ext = format_part.replace("image/", "")
            if ext == "jpeg":
                ext = "jpg"
        else:
            ext = "png"

        # 确保输出路径有正确的扩展名
        if not output_path.endswith(f".{ext}"):
            output_path = f"{output_path}.{ext}"

        # 解码并保存
        image_data = base64.b64decode(base64_data)
        with open(output_path, "wb") as f:
            f.write(image_data)

        return output_path

    async def edit_poster(
        self,
        source_image: str,  # base64 data URI 或文件路径
        prompt: str,
        reference_image: Optional[str] = None,  # base64 data URI 或文件路径
        output_dir: str = "./output",
        filename_prefix: str = "edited"
    ) -> dict:
        """
        编辑海报（全图修改或局部修改）
        全图和局部修改使用相同的逻辑，只是输入图片不同

        Args:
            source_image: 源图片（全图修改时为原图，局部修改时为蒙版合并后的图）
                         可以是 base64 data URI 或文件路径
            prompt: 修改提示词
            reference_image: 参考图片（可选），可以是 base64 data URI 或文件路径
            output_dir: 输出目录
            filename_prefix: 文件名前缀

        Returns:
            包含结果的字典，格式同 PosterGenerator.generate()
        """
        import tempfile
        import shutil

        # 创建临时目录处理图片
        temp_dir = tempfile.mkdtemp()

        try:
            # 处理源图片
            if source_image.startswith("data:"):
                source_path = self.decode_base64_image(
                    source_image,
                    os.path.join(temp_dir, "source")
                )
            else:
                source_path = source_image

            # 处理参考图片
            reference_path = None
            if reference_image:
                if reference_image.startswith("data:"):
                    reference_path = self.decode_base64_image(
                        reference_image,
                        os.path.join(temp_dir, "reference")
                    )
                else:
                    reference_path = reference_image

            # 构建图片路径列表
            image_paths = [source_path]
            if reference_path:
                image_paths.append(reference_path)

            # 调用 PosterGenerator 生成图片
            result = await self.generator.generate(
                text_input=prompt,
                image_paths=image_paths,
                output_dir=output_dir,
                filename_prefix=filename_prefix
            )

            return result

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
