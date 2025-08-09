"""
集合管理API
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional

from src.models.collection import CollectionCreate, CollectionResponse, CollectionUpdate
from src.db.vector_db_service import VectorDBService
from src.core.dependencies import get_vector_db

router = APIRouter()

@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection: CollectionCreate,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    创建新的向量集合
    """
    try:
        result = await vector_db.create_collection(collection)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"创建集合失败: {str(e)}"
        )

@router.get("/", response_model=List[CollectionResponse])
async def list_collections(
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    获取所有集合列表
    """
    try:
        collections = await vector_db.list_collections()
        return collections
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取集合列表失败: {str(e)}"
        )

@router.get("/{collection_name}", response_model=CollectionResponse)
async def get_collection(
    collection_name: str,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    获取指定集合的详细信息
    """
    try:
        collection = await vector_db.get_collection(collection_name)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        return collection
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取集合信息失败: {str(e)}"
        )

@router.delete("/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_name: str,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    删除指定集合
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        await vector_db.delete_collection(collection_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除集合失败: {str(e)}"
        )

@router.patch("/{collection_name}", response_model=CollectionResponse)
async def update_collection(
    collection_name: str,
    update_data: CollectionUpdate,
    vector_db: VectorDBService = Depends(get_vector_db)
):
    """
    更新集合信息
    """
    try:
        exists = await vector_db.collection_exists(collection_name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 '{collection_name}' 不存在"
            )
        updated = await vector_db.update_collection(collection_name, update_data)
        return updated
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新集合失败: {str(e)}"
        )