"""
文本分割工具，用于将长文本分割成适合向量化的小块
"""
from typing import List, Optional, Dict, Any
import re

class TextSplitter:
    """
    文本分割器基类
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n"
    ):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块的重叠字符数
            separator: 分割文本的分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成小块
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        # 按分隔符分割文本
        splits = text.split(self.separator)
        
        # 过滤空字符串
        splits = [s for s in splits if s.strip()]
        
        # 合并小块
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # 如果当前块加上新的分割会超过最大长度，则保存当前块并开始新块
            if current_length + len(split) > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))
                
                # 保留重叠部分
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk) + len(self.separator) * (len(current_chunk) - 1)
            
            # 添加新的分割到当前块
            current_chunk.append(split)
            current_length += len(split) + len(self.separator)
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks

class ChineseTextSplitter(TextSplitter):
    """
    中文文本分割器，针对中文文本特点进行优化
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """
        初始化中文文本分割器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块的重叠字符数
        """
        super().__init__(chunk_size, chunk_overlap, separator="\n")
    
    def split_text(self, text: str) -> List[str]:
        """
        将中文文本分割成小块，优先在段落、句号、逗号处分割
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        # 首先按段落分割
        paragraphs = text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 进一步分割长段落
        splits = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                splits.append(paragraph)
            else:
                # 按句号分割
                sentences = re.split(r'([。！？])', paragraph)
                # 保留分隔符
                sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2] + [""])]
                sentences = [s.strip() for s in sentences if s.strip()]
                
                # 如果句子仍然太长，按逗号分割
                for sentence in sentences:
                    if len(sentence) <= self.chunk_size:
                        splits.append(sentence)
                    else:
                        # 按逗号分割
                        clauses = re.split(r'([，；：、])', sentence)
                        # 保留分隔符
                        clauses = ["".join(i) for i in zip(clauses[0::2], clauses[1::2] + [""])]
                        clauses = [c.strip() for c in clauses if c.strip()]
                        splits.extend(clauses)
        
        # 合并小块
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # 如果单个分割就超过了最大长度，则直接截断
            if len(split) > self.chunk_size:
                # 如果当前块不为空，先保存
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # 直接截断长文本
                for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                    chunks.append(split[i:i + self.chunk_size])
                
                continue
            
            # 如果当前块加上新的分割会超过最大长度，则保存当前块并开始新块
            if current_length + len(split) > self.chunk_size and current_chunk:
                chunks.append("".join(current_chunk))
                
                # 保留重叠部分
                overlap_text = "".join(current_chunk)[-self.chunk_overlap:]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            
            # 添加新的分割到当前块
            current_chunk.append(split)
            current_length += len(split)
        
        # 添加最后一个块
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks