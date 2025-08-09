"""
搜索相关数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union


class SearchQuery(BaseModel):
    """搜索查询模型"""
    text: Optional[str] = Field(None, description="文本查询，将被转换为向量")
    vector: Optional[List[float]] = Field(None, description="查询向量")
    limit: int = Field(10, description="返回结果数量")
    filter: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    with_payload: bool = Field(True, description="是否返回向量附加数据")
    with_vector: bool = Field(False, description="是否返回向量数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "如何使用向量数据库",
                "limit": 5,
                "filter": {
                    "metadata.domain": "技术文档"
                },
                "with_payload": True,
                "with_vector": False
            }
        }


class SearchMatch(BaseModel):
    """搜索匹配结果模型"""
    id: str = Field(..., description="向量ID")
    score: float = Field(..., description="相似度得分")
    payload: Optional[Dict[str, Any]] = Field(None, description="向量附加数据")
    vector: Optional[List[float]] = Field(None, description="向量数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "score": 0.95,
                "payload": {
                    "text": "向量数据库是一种专门设计用于存储、索引和查询向量嵌入的数据库系统。",
                    "source": "技术文档",
                    "metadata": {
                        "author": "张三",
                        "date": "2023-01-01"
                    }
                },
                "vector": None
            }
        }


class SearchResult(BaseModel):
    """搜索结果模型"""
    matches: List[SearchMatch] = Field(..., description="匹配结果列表")
    query: Optional[str] = Field(None, description="原始查询文本")
    
    class Config:
        json_schema_extra = {
            "example": {
                "matches": [
                    {
                        "id": "doc_001",
                        "score": 0.95,
                        "payload": {
                            "text": "向量数据库是一种专门设计用于存储、索引和查询向量嵌入的数据库系统。",
                            "source": "技术文档"
                        },
                        "vector": None
                    },
                    {
                        "id": "doc_002",
                        "score": 0.85,
                        "payload": {
                            "text": "向量数据库通过专门的索引结构解决了高维向量空间中的相似性搜索问题。",
                            "source": "技术文档"
                        },
                        "vector": None
                    }
                ],
                "query": "如何使用向量数据库"
            }
        }


class SearchByTextRequest(BaseModel):
    """文本搜索请求模型"""
    query: str = Field(..., description="查询文本")
    limit: int = Field(10, description="返回结果数量")
    filter: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    with_payload: bool = Field(True, description="是否返回向量附加数据")
    with_vector: bool = Field(False, description="是否返回向量数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "如何使用向量数据库",
                "limit": 5,
                "filter": {
                    "source": "技术文档"
                },
                "with_payload": True,
                "with_vector": False
            }
        }


class SearchByVectorRequest(BaseModel):
    """向量搜索请求模型"""
    vector: List[float] = Field(..., description="查询向量")
    limit: int = Field(10, description="返回结果数量")
    filter: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    with_payload: bool = Field(True, description="是否返回向量附加数据")
    with_vector: bool = Field(False, description="是否返回向量数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vector": [0.1, 0.2, 0.3, 0.4],
                "limit": 5,
                "filter": {
                    "source": "技术文档"
                },
                "with_payload": True,
                "with_vector": False
            }
        }