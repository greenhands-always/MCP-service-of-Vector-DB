"""
依赖注入模块
"""
from fastapi import Depends
from functools import lru_cache

from src.core.config import settings
from src.db.vector_db_service import VectorDBService
from src.core.embedding import EmbeddingService

@lru_cache()
def get_vector_db() -> VectorDBService:
    """
    获取向量数据库服务实例
    """
    return VectorDBService(
        host=settings.VECTOR_DB_HOST,
        port=settings.VECTOR_DB_PORT,
        grpc_port=settings.VECTOR_DB_GRPC_PORT,
        api_key=settings.VECTOR_DB_API_KEY,
        https=settings.VECTOR_DB_HTTPS
    )

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    获取嵌入服务实例
    """
    return EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY,
        api_base=settings.EMBEDDING_API_BASE
    )