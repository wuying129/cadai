#!/usr/bin/env python3
"""
批量处理器
解析设计AI输出并批量生成海报
"""

import re
import json
import os
from typing import Dict, List, Optional


class BatchProcessor:
    """批量处理器"""
    
    def parse_design_output(self, text: str) -> Dict[str, str]:
        """
        解析设计AI输出，提取每个海报的完整原始文本
        
        Args:
            text: 设计AI的原始输出文本
        
        Returns:
            字典格式: {"海报01 - 主KV视觉": "完整原始文本...", ...}
        """
        prompts = {}
        
        # 匹配模式：#### 海报XX - 名称 (English Name) 或 海报XX - 名称
        poster_pattern = r'#{2,4}\s*海报\s*(\d+)\s*[-–—]\s*([^(\n]+)(?:\(([^)]+)\))?'
        
        # 找到所有海报标题
        matches = list(re.finditer(poster_pattern, text))
        
        if not matches:
            print("未找到海报标题，尝试备用模式...")
            # 备用模式：更宽松的匹配
            poster_pattern = r'海报\s*(\d+)\s*[-–—]\s*([^\n(]+)'
            matches = list(re.finditer(poster_pattern, text))
        
        if not matches:
            print("尝试第三种模式...")
            # 第三种模式：匹配 "海报01" 或 "Poster 01"
            poster_pattern = r'(?:海报|Poster)\s*(\d+)[^\n]*[-–—:：]\s*([^\n]+)'
            matches = list(re.finditer(poster_pattern, text, re.IGNORECASE))
        
        print(f"找到 {len(matches)} 个海报")
        
        for i, match in enumerate(matches):
            poster_num = match.group(1).zfill(2)
            poster_name_cn = match.group(2).strip()
            
            # 清理名称
            poster_name_cn = re.sub(r'\s+', ' ', poster_name_cn)
            poster_name_cn = poster_name_cn.rstrip('*#')
            
            # 海报名称作为key
            poster_name = f"海报{poster_num} - {poster_name_cn}"
            
            # 获取该海报的完整内容（从标题开始到下一个海报或文本末尾）
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_pos:end_pos].strip()
            
            prompts[poster_name] = content
            print(f"  ✓ {poster_name}")
        
        return prompts
    
    def extract_prompt_details(self, poster_content: str) -> Dict:
        """
        从海报内容中提取详细信息
        
        Args:
            poster_content: 单个海报的完整文本
        
        Returns:
            包含提示词详情的字典
        """
        result = {
            "prompt_cn": "",
            "prompt_en": "",
            "negative": "",
            "layout": ""
        }
        
        # 提取中文提示词
        cn_patterns = [
            r'提示词[（(]中文[）)][:：]\s*(.*?)(?=(?:Prompt|英文|负面词|Negative|排版|$))',
            r'中文提示词[:：]\s*(.*?)(?=(?:Prompt|英文|负面词|Negative|排版|$))',
            r'【中文提示词】\s*(.*?)(?=(?:【|Prompt|英文|负面词|Negative|$))'
        ]
        
        for pattern in cn_patterns:
            cn_match = re.search(pattern, poster_content, re.DOTALL | re.IGNORECASE)
            if cn_match:
                result["prompt_cn"] = cn_match.group(1).strip()
                break
        
        # 提取英文提示词
        en_patterns = [
            r'(?:Prompt\s*\(?English\)?|英文\s*Prompt|English\s*Prompt)[:：]\s*(.*?)(?=(?:负面词|Negative|排版|布局|$))',
            r'【英文提示词】\s*(.*?)(?=(?:【|负面词|Negative|$))'
        ]
        
        for pattern in en_patterns:
            en_match = re.search(pattern, poster_content, re.DOTALL | re.IGNORECASE)
            if en_match:
                result["prompt_en"] = en_match.group(1).strip()
                break
        
        # 提取负面词
        neg_patterns = [
            r'(?:负面词|Negative)[^:：]*[:：]\s*(.*?)(?=(?:###|海报\s*\d+|排版|布局|$))',
            r'【负面提示词】\s*(.*?)(?=(?:【|$))'
        ]
        
        for pattern in neg_patterns:
            neg_match = re.search(pattern, poster_content, re.DOTALL | re.IGNORECASE)
            if neg_match:
                result["negative"] = neg_match.group(1).strip()
                break
        
        # 提取排版说明
        layout_patterns = [
            r'(?:排版|布局)[^:：]*[:：]\s*(.*?)(?=(?:###|海报\s*\d+|负面词|Negative|$))',
            r'【排版说明】\s*(.*?)(?=(?:【|$))'
        ]
        
        for pattern in layout_patterns:
            layout_match = re.search(pattern, poster_content, re.DOTALL | re.IGNORECASE)
            if layout_match:
                result["layout"] = layout_match.group(1).strip()
                break
        
        return result
    
    def validate_prompts(self, prompts: Dict[str, str]) -> Dict[str, bool]:
        """
        验证提示词的完整性
        
        Args:
            prompts: 解析后的提示词字典
        
        Returns:
            验证结果字典
        """
        validation = {}
        
        for name, content in prompts.items():
            is_valid = True
            issues = []
            
            # 检查内容长度
            if len(content) < 100:
                is_valid = False
                issues.append("内容过短")
            
            # 检查是否包含关键元素
            if "提示词" not in content and "Prompt" not in content.lower():
                issues.append("可能缺少提示词")
            
            validation[name] = {
                "valid": is_valid,
                "issues": issues,
                "content_length": len(content)
            }
        
        return validation
    
    def save_prompts_json(self, prompts: Dict[str, str], output_path: str) -> str:
        """
        保存提示词到JSON文件
        
        Args:
            prompts: 解析后的提示词字典
            output_path: 输出文件路径
        
        Returns:
            保存的文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 提示词已保存: {output_path}")
        return output_path
    
    def get_poster_list(self, prompts: Dict[str, str]) -> List[Dict]:
        """
        获取海报列表（用于前端展示）
        
        Args:
            prompts: 解析后的提示词字典
        
        Returns:
            海报列表
        """
        posters = []
        
        for idx, (name, content) in enumerate(prompts.items()):
            # 从名称中提取编号和标题
            match = re.match(r'海报(\d+)\s*[-–—]\s*(.+)', name)
            if match:
                num = match.group(1)
                title = match.group(2).strip()
            else:
                num = str(idx + 1).zfill(2)
                title = name
            
            posters.append({
                "id": f"poster_{num}",
                "number": num,
                "title": title,
                "full_name": name,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            })
        
        return posters
