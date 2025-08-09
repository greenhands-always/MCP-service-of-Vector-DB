"""
搜索API
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict, Any

from src.models.search import SearchQuery, SearchResult
from src.db.vector_db_service import VectorDBService
from src.core.dependencies import get_vector_db, get_embedding_service
from src.core.embedding import EmbeddingService

router = APIRouter()

@router.post("/{collection_name}", response_model=SearchResult)
async def search_vectors(
    collection_name: str,
    query: SearchQuery,
    vector_db: VectorDBService = Depends(get_vector_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    在指定集合中搜索相似向量
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        
        # 如果提供了文本查询，则生成查询向量
        if query.text and not query.vector:
            query.vector = await embedding_service.embed_text(query.text)
        
        # 如果既没有提供文本查询也没有提供向量，则返回错误
        if not query.vector:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="必须提供文本查询或向量"
            )
        
        result = await vector_db.search(
            collection_name=collection_name,
            query_vector=query.vector,
            limit=query.limit,
            filter_params=query.filter,
            with_payload=query.with_payload,
            with_vector=query.with_vector
        )
        
        return SearchResult(
            matches=result,
            query=query.text if query.text else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索失败: {str(e)}"
        )

@router.post("/{collection_name}/text", response_model=SearchResult)
async def search_by_text(
    collection_name: str,
    text: str,
    limit: int = 10,
    filter_params: Optional[Dict[str, Any]] = None,
    with_payload: bool = True,
    with_vector: bool = False,
    vector_db: VectorDBService = Depends(get_vector_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    使用文本在指定集合中搜索相似向量（简化接口）
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        
        # 生成查询向量
        query_vector = await embedding_service.embed_text(text)
        
        result = await vector_db.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            filter_params=filter_params,
            with_payload=with_payload,
            with_vector=with_vector
        )
        
        return SearchResult(
            matches=result,
            query=text
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索失败: {str(e)}"
        )