#!/usr/bin/env python3
"""存储管理模块"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List


class StorageManager:
    """存储管理器"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.outputs_dir = self.data_dir / "outputs"
        self.designs_dir = self.data_dir / "designs"
        self.prompts_dir = self.data_dir / "prompts"
        
        for d in [self.uploads_dir, self.outputs_dir, self.designs_dir, self.prompts_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_output_dir(self, user_id: str, product_id: str) -> Path:
        d = self.outputs_dir / user_id / product_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def save_design_output(self, user_id: str, product_id: str, content: str) -> str:
        d = self.designs_dir / user_id / product_id
        d.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = d / f"design_output_{ts}.txt"
        path.write_text(content, encoding='utf-8')
        
        latest = d / "design_output_latest.txt"
        latest.write_text(content, encoding='utf-8')
        
        return str(path)
    
    def save_prompts(self, user_id: str, product_id: str, prompts: dict) -> str:
        d = self.prompts_dir / user_id / product_id
        d.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = d / f"prompts_{ts}.json"
        path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding='utf-8')
        
        latest = d / "prompts_latest.json"
        latest.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding='utf-8')
        
        return str(path)
    
    def save_summary(self, user_id: str, product_id: str, summary: dict) -> str:
        d = self.get_output_dir(user_id, product_id)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = d / f"summary_{ts}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        
        latest = d / "summary_latest.json"
        latest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        
        return str(path)
    
    def list_uploaded_images(self, user_id: str, product_id: str) -> List[str]:
        d = self.uploads_dir / user_id / product_id
        if not d.exists():
            return []
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            images.extend([str(p) for p in d.glob(ext)])
        return sorted(images)
    
    def list_generated_posters(self, user_id: str, product_id: str) -> List[dict]:
        d = self.get_output_dir(user_id, product_id)
        posters = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            for p in d.glob(ext):
                posters.append({
                    "name": p.stem,
                    "filename": p.name,
                    "path": str(p),
                    "size": p.stat().st_size
                })
        return sorted(posters, key=lambda x: x['name'])


class TaskDatabase:
    """任务数据库（简单JSON实现）"""
    
    def __init__(self, data_dir: Path):
        self.db_file = Path(data_dir) / "tasks_db.json"
        self._load()
    
    def _load(self):
        if self.db_file.exists():
            self.data = json.loads(self.db_file.read_text(encoding='utf-8'))
        else:
            self.data = {"tasks": {}}
    
    def _save(self):
        self.db_file.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf-8')
    
    def create_task(self, task_id: str, task_data: dict):
        self.data["tasks"][task_id] = task_data
        self._save()
    
    def update_task(self, task_id: str, updates: dict):
        if task_id in self.data["tasks"]:
            self.data["tasks"][task_id].update(updates)
            self._save()
    
    def get_task(self, task_id: str) -> Optional[dict]:
        return self.data["tasks"].get(task_id)
    
    def delete_task(self, task_id: str):
        if task_id in self.data["tasks"]:
            del self.data["tasks"][task_id]
            self._save()
