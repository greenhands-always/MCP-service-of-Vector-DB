"""
向量管理API
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict, Any

from src.models.vector import VectorUpload, VectorResponse, VectorBatchUpload, VectorBatchResponse
from src.db.vector_db_service import VectorDBService
from src.core.dependencies import get_vector_db

router = APIRouter()

@router.post("/{collection_name}", response_model=VectorResponse, status_code=status.HTTP_201_CREATED)
async def upload_vector(
    collection_name: str,
    vector_data: VectorUpload,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    上传单个向量到指定集合
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        
        result = await vector_db.add_vector(collection_name, vector_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"上传向量失败: {str(e)}"
        )

@router.post("/{collection_name}/batch", response_model=VectorBatchResponse, status_code=status.HTTP_201_CREATED)
async def upload_vectors_batch(
    collection_name: str,
    batch_data: VectorBatchUpload,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    批量上传向量到指定集合
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        
        result = await vector_db.add_vectors_batch(collection_name, batch_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"批量上传向量失败: {str(e)}"
        )

@router.get("/{collection_name}/{vector_id}", response_model=VectorResponse)
async def get_vector(
    collection_name: str,
    vector_id: str,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    获取指定向量的详细信息
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        
        vector = await vector_db.get_vector(collection_name, vector_id)
        if not vector:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"向量 ID '{vector_id}' 不存在"
            )
        return vector
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取向量失败: {str(e)}"
        )

@router.delete("/{collection_name}/{vector_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vector(
    collection_name: str,
    vector_id: str,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    删除指定向量
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        
        vector_exists = await vector_db.vector_exists(collection_name, vector_id)
        if not vector_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"向量 ID '{vector_id}' 不存在"
            )
        
        await vector_db.delete_vector(collection_name, vector_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除向量失败: {str(e)}"
        )

@router.delete("/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vectors(
    collection_name: str,
    filter_params: Optional[Dict[str, Any]] = None,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    根据过滤条件删除向量
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        
        await vector_db.delete_vectors(collection_name, filter_params)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除向量失败: {str(e)}"
        )