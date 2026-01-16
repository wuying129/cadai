#!/usr/bin/env python3
"""日志工具模块"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class TaskLogger:
    """任务专属日志记录器"""
    
    def __init__(self, task_id: str, logs_dir: Path):
        self.task_id = task_id
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"task_{task_id}_{timestamp}.log"
        self.file = open(self.log_file, 'w', encoding='utf-8')
        
        self.file.write("=" * 80 + "\n")
        self.file.write(f"任务ID: {task_id}\n")
        self.file.write(f"开始时间: {datetime.now().isoformat()}\n")
        self.file.write("=" * 80 + "\n\n")
        self.file.flush()
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}\n"
        self.file.write(line)
        self.file.flush()
        print(f"[Task:{self.task_id}] {message}")
    
    def log_dict(self, data: dict, title: str = "Data"):
        import json
        self.log(f"\n--- {title} ---")
        self.file.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        self.file.write("--- End ---\n\n")
        self.file.flush()
    
    def close(self):
        self.file.write("\n" + "=" * 80 + "\n")
        self.file.write(f"结束时间: {datetime.now().isoformat()}\n")
        self.file.write("=" * 80 + "\n")
        self.file.close()


class StageLogger:
    """阶段日志记录器"""
    
    def __init__(self, task_logger: TaskLogger):
        self.task_logger = task_logger
        self.current_stage = None
        self.stage_start = None
    
    def start_stage(self, name: str):
        self.current_stage = name
        self.stage_start = datetime.now()
        self.task_logger.log(f"\n{'='*20} 阶段: {name} {'='*20}")
    
    def end_stage(self, success: bool = True):
        if not self.current_stage:
            return
        duration = (datetime.now() - self.stage_start).total_seconds()
        status = "✓ 成功" if success else "✗ 失败"
        self.task_logger.log(f"{'='*20} {status} | 耗时: {duration:.2f}s {'='*20}\n")
        self.current_stage = None
