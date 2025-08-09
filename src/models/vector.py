"""
向量相关数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union


class VectorUpload(BaseModel):
    """上传单个向量的请求模型"""
    id: Optional[str] = Field(None, description="向量ID，如不提供则自动生成")
    vector: List[float] = Field(..., description="向量数据")
    payload: Optional[Dict[str, Any]] = Field(None, description="向量附加数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "text": "这是一个示例文档",
                    "source": "示例数据库",
                    "metadata": {
                        "author": "张三",
                        "date": "2023-01-01"
                    }
                }
            }
        }


class VectorBatchUpload(BaseModel):
    """批量上传向量的请求模型"""
    vectors: List[VectorUpload] = Field(..., description="向量列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vectors": [
                    {
                        "id": "doc_001",
                        "vector": [0.1, 0.2, 0.3, 0.4],
                        "payload": {
                            "text": "这是第一个示例文档",
                            "source": "示例数据库"
                        }
                    },
                    {
                        "id": "doc_002",
                        "vector": [0.2, 0.3, 0.4, 0.5],
                        "payload": {
                            "text": "这是第二个示例文档",
                            "source": "示例数据库"
                        }
                    }
                ]
            }
        }


class VectorResponse(BaseModel):
    """向量响应模型"""
    id: str = Field(..., description="向量ID")
    vector: Optional[List[float]] = Field(None, description="向量数据")
    payload: Optional[Dict[str, Any]] = Field(None, description="向量附加数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "text": "这是一个示例文档",
                    "source": "示例数据库",
                    "metadata": {
                        "author": "张三",
                        "date": "2023-01-01"
                    }
                }
            }
        }


class VectorBatchResponse(BaseModel):
    """批量向量操作响应模型"""
    ids: List[str] = Field(..., description="操作成功的向量ID列表")
    status: str = Field("success", description="操作状态")
    added_count: int = Field(0, description="添加数量")
    error_count: int = Field(0, description="错误数量")
    errors: Optional[Dict[str, str]] = Field(None, description="错误详情")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ids": ["doc_001", "doc_002"],
                "status": "success",
                "added_count": 2,
                "error_count": 0,
                "errors": None
            }
        }


class VectorDeleteResponse(BaseModel):
    """向量删除响应模型"""
    deleted_count: int = Field(..., description="删除的向量数量")
    status: str = Field("success", description="操作状态")
    errors: Optional[List[str]] = Field(None, description="错误信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "deleted_count": 1,
                "status": "success",
                "errors": None
            }
        }