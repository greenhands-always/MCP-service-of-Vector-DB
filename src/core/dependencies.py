"""
依赖注入模块
"""
from fastapi import Depends
from functools import lru_cache

from src.core.config import settings
from src.db.vector_db_service import VectorDBService
from src.core.embedding import EmbeddingService, create_embedding_service


@lru_cache()
def get_vector_db() -> VectorDBService:
    """
    获取向量数据库服务实例
    """
    return VectorDBService()

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    获取嵌入服务实例
    """
    return create_embedding_service()

