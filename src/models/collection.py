"""
集合相关数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class CollectionCreate(BaseModel):
    """创建集合的请求模型"""
    name: str = Field(..., description="集合名称")
    vector_size: int = Field(..., description="向量维度")
    description: Optional[str] = Field(None, description="集合描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="集合元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "documents",
                "vector_size": 768,
                "description": "文档向量集合",
                "metadata": {
                    "domain": "general",
                    "language": "zh"
                }
            }
        }


class CollectionInfo(BaseModel):
    """集合信息模型"""
    name: str = Field(..., description="集合名称")
    vector_size: int = Field(..., description="向量维度")
    vector_count: int = Field(0, description="向量数量")
    description: Optional[str] = Field(None, description="集合描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="集合元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "documents",
                "vector_size": 768,
                "vector_count": 1000,
                "description": "文档向量集合",
                "metadata": {
                    "domain": "general",
                    "language": "zh"
                }
            }
        }


class CollectionList(BaseModel):
    """集合列表模型"""
    collections: List[CollectionInfo] = Field(..., description="集合列表")
    total: int = Field(..., description="总集合数")
    limit: int = Field(..., description="返回结果数量限制")
    offset: int = Field(..., description="结果偏移量")
    
    class Config:
        json_schema_extra = {
            "example": {
                "collections": [
                    {
                        "name": "documents",
                        "vector_size": 768,
                        "vector_count": 1000,
                        "description": "文档向量集合",
                        "metadata": {
                            "domain": "general",
                            "language": "zh"
                        }
                    }
                ],
                "total": 1,
                "limit": 100,
                "offset": 0
            }
        }


class CollectionResponse(BaseModel):
    """集合响应模型"""
    name: str = Field(..., description="集合名称")
    vector_size: int = Field(..., description="向量维度")
    vector_count: int = Field(0, description="向量数量")
    description: Optional[str] = Field(None, description="集合描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="集合元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "documents",
                "vector_size": 768,
                "vector_count": 1000,
                "description": "文档向量集合",
                "metadata": {
                    "domain": "general",
                    "language": "zh"
                }
            }
        }


class CollectionUpdate(BaseModel):
    """更新集合的请求模型"""
    description: Optional[str] = Field(None, description="集合描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="集合元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "更新后的文档向量集合",
                "metadata": {
                    "domain": "updated",
                    "language": "zh"
                }
            }
        }


class CollectionDeleteResponse(BaseModel):
    """集合删除响应模型"""
    name: str = Field(..., description="集合名称")
    status: str = Field(..., description="删除状态")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "documents",
                "status": "success"
            }
        }
