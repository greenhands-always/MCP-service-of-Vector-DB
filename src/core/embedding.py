"""
嵌入服务模块，用于文本向量化
"""
from typing import List, Union, Dict, Any
import numpy as np
from loguru import logger

from src.core.config import settings

class EmbeddingService:
    """
    嵌入服务类，用于将文本转换为向量
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化嵌入服务
        
        Args:
            model_name: 模型名称，默认使用配置中的EMBEDDING_MODEL
            device: 设备，默认使用配置中的EMBEDDING_DEVICE
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        
        logger.info(f"使用模拟嵌入服务: {self.model_name}")
        self.vector_size = 768  # 使用固定的向量维度
        logger.info(f"模拟嵌入服务初始化完成，向量维度: {self.vector_size}")
    
    async def get_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            嵌入向量或向量列表
        """
        if isinstance(text, str):
            # 单个文本，生成固定的向量
            embedding = [0.1] * self.vector_size
            return embedding
        else:
            # 文本列表，为每个文本生成固定的向量
            embeddings = []
            for i, t in enumerate(text):
                # 生成稍微不同的向量
                embedding = [0.1 * (i + 1) + 0.01 * j for j in range(self.vector_size)]
                embeddings.append(embedding)
            return embeddings
    
    async def get_embeddings_with_metadata(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        获取文本的嵌入向量，并附加元数据
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表，与texts一一对应
            
        Returns:
            包含向量和元数据的字典
        """
        embeddings = await self.get_embeddings(texts)
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        return {
            "embeddings": embeddings,
            "metadatas": metadatas,
            "texts": texts,
            "model": self.model_name,
            "dimension": self.vector_size
        }