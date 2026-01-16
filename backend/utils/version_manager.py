#!/usr/bin/env python3
"""
版本历史管理器
管理海报的多个编辑版本
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


class VersionManager:
    """版本历史管理器"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.versions_dir = self.data_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def get_poster_dir(self, user_id: str, product_id: str, poster_id: str) -> Path:
        """获取海报版本目录"""
        d = self.versions_dir / user_id / product_id / poster_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_version_index(self, poster_dir: Path) -> dict:
        """加载版本索引文件"""
        index_file = poster_dir / "versions.json"
        if index_file.exists():
            return json.loads(index_file.read_text(encoding='utf-8'))
        return self._create_empty_index(poster_dir)

    def _create_empty_index(self, poster_dir: Path) -> dict:
        """创建空版本索引"""
        return {
            "poster_id": poster_dir.name,
            "current_version": 0,
            "total_versions": 0,
            "versions": []
        }

    def create_version(
        self,
        user_id: str,
        product_id: str,
        poster_id: str,
        version_type: str,  # initial, full_edit, partial_edit
        original_image: str,
        generated_image: str,
        prompt: str,
        reference_image: Optional[str] = None,
        parent_version: Optional[int] = None
    ) -> int:
        """
        创建新版本

        Args:
            user_id: 用户ID
            product_id: 产品ID
            poster_id: 海报ID
            version_type: 版本类型
            original_image: 原始图片路径
            generated_image: 生成图片路径
            prompt: 编辑提示词
            reference_image: 参考图片路径（可选）
            parent_version: 父版本号（可选）

        Returns:
            新版本号
        """
        poster_dir = self.get_poster_dir(user_id, product_id, poster_id)
        index = self.load_version_index(poster_dir)

        # 新版本号
        new_version = index["current_version"] + 1
        version_dir = poster_dir / f"v{new_version:02d}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # 复制/保存文件
        shutil.copy(original_image, version_dir / "original.png")
        shutil.copy(generated_image, version_dir / f"{poster_id}_v{new_version:02d}.png")

        if reference_image:
            shutil.copy(reference_image, version_dir / "reference.png")

        # 创建元数据
        meta = {
            "version": new_version,
            "type": version_type,
            "created_at": datetime.now().isoformat(),
            "parent_version": parent_version,
            "edit_request": {
                "prompt": prompt,
                "has_reference": reference_image is not None
            }
        }

        (version_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        # 更新索引
        version_entry = {
            "version": new_version,
            "type": version_type,
            "created_at": meta["created_at"],
            "path": f"v{new_version:02d}",
            "filename": f"{poster_id}_v{new_version:02d}.png",
            "prompt": prompt,
            "has_reference": reference_image is not None
        }

        # 初始版本没有 prompt
        if version_type == "initial":
            version_entry.pop("prompt", None)

        index["versions"].append(version_entry)
        index["current_version"] = new_version
        index["total_versions"] = len(index["versions"])

        # 保存索引
        (poster_dir / "versions.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        return new_version

    def get_versions(
        self,
        user_id: str,
        product_id: str,
        poster_id: str,
        base_url_prefix: str = ""
    ) -> dict:
        """
        获取版本列表

        Args:
            user_id: 用户ID
            product_id: 产品ID
            poster_id: 海报ID
            base_url_prefix: URL前缀（用于构建图片URL）

        Returns:
            版本索引字典，包含完整的URL
        """
        poster_dir = self.get_poster_dir(user_id, product_id, poster_id)
        index = self.load_version_index(poster_dir)

        # 添加URL
        if base_url_prefix:
            for v in index["versions"]:
                v["thumbnail_url"] = f"{base_url_prefix}/v{v['version']:02d}/thumbnail"
                v["full_url"] = f"{base_url_prefix}/v{v['version']:02d}/full"

        return index

    def get_version_image(
        self,
        user_id: str,
        product_id: str,
        poster_id: str,
        version: int,
        size: str = "full"
    ) -> Optional[str]:
        """
        获取版本图片路径

        Args:
            user_id: 用户ID
            product_id: 产品ID
            poster_id: 海报ID
            version: 版本号
            size: 图片大小 (full 或 thumbnail)

        Returns:
            图片文件路径，如果不存在则返回 None
        """
        poster_dir = self.get_poster_dir(user_id, product_id, poster_id)
        version_dir = poster_dir / f"v{version:02d}"

        if not version_dir.exists():
            return None

        if size == "thumbnail":
            # 缩略图使用生成的图片
            thumbnail_path = version_dir / f"{poster_id}_v{version:02d}.png"
            if thumbnail_path.exists():
                return str(thumbnail_path)
        else:
            image_path = version_dir / f"{poster_id}_v{version:02d}.png"
            if image_path.exists():
                return str(image_path)

        return None

    def initialize_from_poster(
        self,
        user_id: str,
        product_id: str,
        poster_id: str,
        poster_path: str,
        poster_name: str
    ) -> int:
        """
        从现有海报初始化版本历史

        Args:
            user_id: 用户ID
            product_id: 产品ID
            poster_id: 海报ID
            poster_path: 海报文件路径
            poster_name: 海报名称

        Returns:
            初始版本号（总是1）
        """
        poster_dir = self.get_poster_dir(user_id, product_id, poster_id)
        index = self.load_version_index(poster_dir)

        # 如果已有版本，不重复初始化
        if index["total_versions"] > 0:
            return index["current_version"]

        # 创建版本1目录
        version_dir = poster_dir / "v01"
        version_dir.mkdir(parents=True, exist_ok=True)

        # 复制海报作为原始版本
        shutil.copy(poster_path, version_dir / "original.png")
        shutil.copy(poster_path, version_dir / f"{poster_id}_v01.png")

        # 创建元数据
        meta = {
            "version": 1,
            "type": "initial",
            "created_at": datetime.now().isoformat(),
            "poster_name": poster_name
        }

        (version_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        # 更新索引
        index["versions"].append({
            "version": 1,
            "type": "initial",
            "created_at": meta["created_at"],
            "path": "v01",
            "filename": f"{poster_id}_v01.png"
        })
        index["current_version"] = 1
        index["total_versions"] = 1

        (poster_dir / "versions.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        return 1
