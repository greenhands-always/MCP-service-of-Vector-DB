"""
配置模块，用于管理服务配置
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings

# 继承 Pydantic 的 BaseSettings，会带来以下特性:
# 自动读取环境变量
# 字段名会对应同名的大写环境变量
# 支持从 .env 文件加载（可在内部类 Config 指定 env_file）
# 读取顺序：显式传参 > 环境变量 > .env > 字段默认值
class Settings(BaseSettings):
    """
    应用配置类
    """
    # 项目信息
    PROJECT_NAME: str = "向量数据库MCP服务"
    PROJECT_DESCRIPTION: str = "将Qdrant向量数据库转换为MCP服务，用于支持RAG应用开发"
    PROJECT_VERSION: str = "0.1.0"
    
    # API配置
    API_PREFIX: str = "/api/v1"
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Qdrant配置
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: Optional[int] = 6334
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_HTTPS: bool = False
    
    # 向量数据库配置 (Qdrant)
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 8000
    VECTOR_DB_GRPC_PORT: int = 6334
    VECTOR_DB_API_KEY: Optional[str] = None
    VECTOR_DB_HTTPS: bool = False
    
    # 嵌入模型配置
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_API_KEY: str = "your-api-key"
    EMBEDDING_API_BASE: str = "https://api.openai.com/v1"
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    
    # MCP服务器配置
    MCP_SERVER_ENABLED: bool = True
    MCP_SERVER_PATH: str = "/mcp"
    MCP_SERVER_NAME: str = "向量数据库MCP服务"
    MCP_SERVER_DESCRIPTION: str = "提供向量存储和检索功能的MCP服务"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 创建全局设置对象
settings = Settings()