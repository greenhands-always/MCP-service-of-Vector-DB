"""
API路由配置
"""
from fastapi import APIRouter

from src.api.v1 import collections, vectors, search

# 创建API路由
api_router = APIRouter()

# 注册子路由
api_router.include_router(collections.router, prefix="/collections", tags=["collections"])
api_router.include_router(vectors.router, prefix="/vectors", tags=["vectors"])
api_router.include_router(search.router, prefix="/search", tags=["search"])