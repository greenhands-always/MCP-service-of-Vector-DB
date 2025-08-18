"""
向量数据库服务主入口 - MCP服务器版本
"""
import os
import uvicorn
from fastapi import FastAPI
from loguru import logger

from src.api.router import api_router
from src.core.config import settings
from src.core.dependencies import get_vector_db, get_embedding_service
from src.mcp.server import MCPServer

# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 注册API路由
app.include_router(api_router, prefix=settings.API_PREFIX)

# 创建MCP服务器
vector_db_service = get_vector_db()
embedding_service = get_embedding_service()
mcp_server = MCPServer(vector_db_service, embedding_service)

@app.get("/")
async def root():
    """
    根路径，返回服务基本信息
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "description": settings.PROJECT_DESCRIPTION,
        "status": "running",
        "mcp_endpoint": "/mcp"
    }

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"}

def start():
    """
    启动服务
    """
    logger.info(f"启动 {settings.PROJECT_NAME} MCP服务，版本 {settings.PROJECT_VERSION}")
    logger.info(f"MCP WebSocket端点: ws://{settings.HOST}:{settings.PORT}/mcp")
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )

if __name__ == "__main__":
    start()
