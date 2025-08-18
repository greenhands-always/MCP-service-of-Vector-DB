"""
向量数据库服务实现，基于Qdrant
"""
from typing import List, Dict, Any, Optional, Union
import uuid
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings
from src.models.collection import (
    CollectionCreate, 
    CollectionInfo, 
    CollectionList,
    CollectionDeleteResponse
)
from src.models.vector import (
    VectorUpload, 
    VectorBatchUpload, 
    VectorResponse, 
    VectorBatchResponse,
    VectorDeleteResponse
)
from src.models.search import (
    SearchResult,
    SearchMatch,
    SearchByTextRequest,
    SearchByVectorRequest
)

class VectorDBService:
    """
    向量数据库服务类，封装Qdrant客户端
    """
    
    def __init__(self):
        """
        初始化向量数据库服务
        """
        # 创建Qdrant客户端
        # 尝试使用内存模式
        try:
            if settings.VECTOR_DB_HTTPS:
                self.client = QdrantClient(
                    url=settings.VECTOR_DB_URL,
                    api_key=settings.VECTOR_DB_API_KEY,
                    https=settings.VECTOR_DB_HTTPS,
                )
            else:
                self.client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    grpc_port=settings.QDRANT_GRPC_PORT,
                    api_key=settings.QDRANT_API_KEY,
                    https=settings.QDRANT_HTTPS,
                    timeout=10.0,
                    prefer_grpc=False
                )
            # 测试连接
            self.client.get_collections()
        except Exception as e:
            logger.warning(f"无法连接到远程Qdrant服务器: {e}")
            logger.info("使用磁盘模式访问Qdrant")
            self.client = QdrantClient(path="./qdrant_data",port=settings.QDRANT_PORT)
        logger.info(f"已连接到Qdrant服务器: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    
    async def create_collection(self, collection: CollectionCreate) -> CollectionInfo:
        """
        创建集合
        
        Args:
            collection: 集合创建参数
            
        Returns:
            集合信息
        """
        try:
            # 检查集合是否已存在
            try:
                self.client.get_collection(collection.name)
                logger.warning(f"集合 {collection.name} 已存在")
                # 返回现有集合信息
                return await self.get_collection(collection.name)
            except (UnexpectedResponse, ValueError):
                # 集合不存在，创建新集合
                self.client.create_collection(
                    collection_name=collection.name,
                    vectors_config=qdrant_models.VectorParams(
                        size=collection.vector_size,
                        distance=qdrant_models.Distance.COSINE
                    ),
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        indexing_threshold=20000
                    )
                )
                
                # 为常用的payload字段创建索引
                try:
                    # 为source字段创建关键词索引
                    self.client.create_payload_index(
                        collection_name=collection.name,
                        field_name="source",
                        field_schema=qdrant_models.KeywordIndexParams(
                            type="keyword"
                        )
                    )
                    # 为text字段创建文本索引
                    self.client.create_payload_index(
                        collection_name=collection.name,
                        field_name="text",
                        field_schema=qdrant_models.TextIndexParams(
                            type="text"
                        )
                    )
                    logger.info(f"已为集合 {collection.name} 创建payload索引")
                except Exception as index_error:
                    logger.warning(f"创建payload索引失败: {index_error}")
                logger.info(f"已创建集合: {collection.name}")
                
                # 返回新集合信息
                return await self.get_collection(collection.name)
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            raise
    
    async def get_collection(self, name: str) -> CollectionInfo:
        """
        获取集合信息
        
        Args:
            name: 集合名称
            
        Returns:
            集合信息
        """
        try:
            collection_info = self.client.get_collection(name)
            
            # 获取向量数量
            collection_count = self.client.count(
                collection_name=name,
                exact=True
            )
            
            # 处理不同版本的Qdrant客户端返回的CollectionInfo对象
            if hasattr(collection_info, 'metadata'):
                description = collection_info.metadata.get("description", "") if collection_info.metadata else ""
            else:
                description = ""
            
            return CollectionInfo(
                name=name,
                vector_size=collection_info.config.params.vectors.size,
                vector_count=collection_count.count,
                description=description
            )
        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            raise
    
    async def list_collections(self, limit: int = 100, offset: int = 0) -> CollectionList:
        """
        获取集合列表
        
        Args:
            limit: 返回结果数量限制
            offset: 结果偏移量
            
        Returns:
            集合列表
        """
        try:
            collections_info = self.client.get_collections()
            
            # 获取每个集合的详细信息
            collections = []
            for collection_name in collections_info.collections:
                try:
                    collection_info = await self.get_collection(collection_name)
                    collections.append(collection_info)
                except Exception as e:
                    logger.warning(f"获取集合 {collection_name} 详细信息失败: {str(e)}")
            
            # 应用分页
            paginated_collections = collections[offset:offset+limit]
            
            return CollectionList(
                collections=paginated_collections,
                total=len(collections),
                limit=limit,
                offset=offset
            )
        except Exception as e:
            logger.error(f"获取集合列表失败: {str(e)}")
            raise
    
    async def delete_collection(self, name: str) -> CollectionDeleteResponse:
        """
        删除集合
        
        Args:
            name: 集合名称
            
        Returns:
            删除结果
        """
        try:
            self.client.delete_collection(name)
            logger.info(f"已删除集合: {name}")
            return CollectionDeleteResponse(
                name=name,
                status="success"
            )
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise
    
    async def add_vector(self, collection_name: str, vector: VectorUpload) -> VectorResponse:
        """
        添加单个向量
        
        Args:
            collection_name: 集合名称
            vector: 向量数据
            
        Returns:
            添加结果
        """
        try:
            # 生成ID（如果未提供）
            vector_id = vector.id or str(uuid.uuid4())
            
            # 添加向量
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=vector_id,
                        vector=vector.vector,
                        payload=vector.payload
                    )
                ]
            )
            
            logger.debug(f"已添加向量到集合 {collection_name}: {vector_id}")
            return VectorResponse(
                id=vector_id,
                status="success"
            )
        except Exception as e:
            logger.error(f"添加向量失败: {str(e)}")
            raise
    
    async def add_vectors(self, collection_name: str, batch: VectorBatchUpload) -> VectorBatchResponse:
        """
        批量添加向量
        
        Args:
            collection_name: 集合名称
            batch: 向量批次数据
            
        Returns:
            添加结果
        """
        try:
            # 准备点数据
            points = []
            ids = []
            
            for vector in batch.vectors:
                # 生成ID（如果未提供）
                vector_id = vector.id or str(uuid.uuid4())
                ids.append(vector_id)
                
                points.append(
                    qdrant_models.PointStruct(
                        id=vector_id,
                        vector=vector.vector,
                        payload=vector.payload
                    )
                )
            
            # 批量添加向量
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"已批量添加 {len(points)} 个向量到集合 {collection_name}")
            return VectorBatchResponse(
                ids=ids,
                status="success",
                added_count=len(points)
            )
        except Exception as e:
            logger.error(f"批量添加向量失败: {str(e)}")
            raise
    
    async def delete_vector(self, collection_name: str, vector_id: str) -> VectorDeleteResponse:
        """
        删除向量
        
        Args:
            collection_name: 集合名称
            vector_id: 向量ID
            
        Returns:
            删除结果
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[vector_id]
                )
            )
            
            logger.debug(f"已从集合 {collection_name} 删除向量: {vector_id}")
            return VectorDeleteResponse(
                id=vector_id,
                status="success"
            )
        except Exception as e:
            logger.error(f"删除向量失败: {str(e)}")
            raise
    
    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> VectorDeleteResponse:
        """
        批量删除向量
        
        Args:
            collection_name: 集合名称
            vector_ids: 向量ID列表
            
        Returns:
            删除结果
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=vector_ids
                )
            )
            
            logger.info(f"已从集合 {collection_name} 批量删除 {len(vector_ids)} 个向量")
            return VectorDeleteResponse(
                ids=vector_ids,
                status="success",
                deleted_count=len(vector_ids)
            )
        except Exception as e:
            logger.error(f"批量删除向量失败: {str(e)}")
            raise
    
    async def search_by_vector(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vector: bool = False
    ) -> List[SearchResult]:
        """
        通过向量搜索
        
        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            limit: 返回结果数量
            filter: 过滤条件
            with_payload: 是否返回payload
            with_vector: 是否返回向量
            
        Returns:
            搜索结果列表
        """
        try:
            # 构建过滤条件
            filter_condition = None
            if filter:
                filter_condition = self._build_filter(filter)
            
            # 执行搜索 - 使用search方法
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_condition,
                with_payload=with_payload,
                with_vectors=with_vector
            )
            
            # 转换结果
            matches = []
            for result in search_results:
                search_match = SearchMatch(
                    id=result.id,
                    score=result.score,
                    payload=result.payload if with_payload else None,
                    vector=result.vector if with_vector else None
                )
                matches.append(search_match)
            
            logger.debug(f"在集合 {collection_name} 中搜索到 {len(matches)} 个结果")
            return matches
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            raise
    
    def _build_filter(self, filter_dict: Dict[str, Any]) -> qdrant_models.Filter:
        """
        构建Qdrant过滤条件
        
        Args:
            filter_dict: 过滤条件字典
            
        Returns:
            Qdrant过滤条件
        """
        # 简单实现，仅支持精确匹配
        must_conditions = []
        
        for key, value in filter_dict.items():
            if isinstance(value, list):
                # 列表值，使用OR条件
                should_conditions = []
                for v in value:
                    should_conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=v)
                        )
                    )
                must_conditions.append(
                    qdrant_models.Filter(
                        should=should_conditions
                    )
                )
            else:
                # 单值，使用精确匹配
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value)
                    )
                )
        
        return qdrant_models.Filter(
            must=must_conditions
        )