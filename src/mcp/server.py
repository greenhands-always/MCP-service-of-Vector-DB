"""
MCP服务器实现，将Qdrant向量数据库转换为MCP服务
"""
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Awaitable
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from src.db.vector_db_service import VectorDBService
from src.core.embedding import EmbeddingService
from src.models.search import SearchResult

class MCPServer:
    """
    MCP服务器类，实现Model Context Protocol
    """
    
    def __init__(self, vector_db_service: VectorDBService, embedding_service: EmbeddingService):
        """
        初始化MCP服务器
        
        Args:
            vector_db_service: 向量数据库服务
            embedding_service: 嵌入服务
        """
        self.vector_db_service = vector_db_service
        self.embedding_service = embedding_service
        self.active_connections: Dict[str, WebSocket] = {}
        self.server_info = {
            "name": "vector-db-mcp",
            "display_name": "向量数据库MCP服务",
            "description": "提供向量存储和语义搜索功能的MCP服务",
            "version": "0.1.0",
            "tools": self._get_tools_schema(),
            "resources": self._get_resources_schema()
        }
        logger.info("MCP服务器初始化完成")
    
    async def handle_connection(self, websocket: WebSocket):
        """
        处理WebSocket连接
        
        Args:
            websocket: WebSocket连接
        """
        # 接受连接
        await websocket.accept()
        
        # 生成连接ID
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        logger.info(f"MCP客户端已连接: {connection_id}")
        
        try:
            # 发送服务器信息
            await self._send_message(websocket, {
                "type": "server_info",
                "data": self.server_info
            })
            
            # 处理消息
            while True:
                # 接收消息
                message_text = await websocket.receive_text()
                message = json.loads(message_text)
                
                # 处理消息
                await self._handle_message(websocket, message)
                
        except WebSocketDisconnect:
            # 客户端断开连接
            logger.info(f"MCP客户端已断开连接: {connection_id}")
            self.active_connections.pop(connection_id, None)
        except Exception as e:
            # 处理其他异常
            logger.error(f"处理MCP连接时发生错误: {str(e)}")
            self.active_connections.pop(connection_id, None)
    
    async def _handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        处理客户端消息
        
        Args:
            websocket: WebSocket连接
            message: 客户端消息
        """
        message_type = message.get("type")
        
        if message_type == "tool_call":
            # 处理工具调用
            await self._handle_tool_call(websocket, message)
        elif message_type == "resource_access":
            # 处理资源访问
            await self._handle_resource_access(websocket, message)
        else:
            # 未知消息类型
            logger.warning(f"收到未知类型的消息: {message_type}")
            await self._send_error(websocket, "unknown_message_type", f"未知的消息类型: {message_type}")
    
    async def _handle_tool_call(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        处理工具调用
        
        Args:
            websocket: WebSocket连接
            message: 工具调用消息
        """
        call_id = message.get("call_id")
        tool_name = message.get("tool_name")
        arguments = message.get("arguments", {})
        
        logger.debug(f"处理工具调用: {tool_name}, 参数: {arguments}")
        
        # 工具映射
        tool_handlers = {
            "create_collection": self._tool_create_collection,
            "list_collections": self._tool_list_collections,
            "delete_collection": self._tool_delete_collection,
            "add_vectors": self._tool_add_vectors,
            "search_by_text": self._tool_search_by_text,
            "search_by_vector": self._tool_search_by_vector
        }
        
        # 检查工具是否存在
        if tool_name not in tool_handlers:
            await self._send_error(websocket, "unknown_tool", f"未知的工具: {tool_name}", call_id=call_id)
            return
        
        try:
            # 调用对应的工具处理函数
            handler = tool_handlers[tool_name]
            result = await handler(arguments)
            
            # 发送成功响应
            await self._send_message(websocket, {
                "type": "tool_response",
                "call_id": call_id,
                "status": "success",
                "result": result
            })
        except Exception as e:
            # 发送错误响应
            logger.error(f"工具调用失败: {str(e)}")
            await self._send_error(websocket, "tool_execution_error", str(e), call_id=call_id)
    
    async def _handle_resource_access(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        处理资源访问
        
        Args:
            websocket: WebSocket连接
            message: 资源访问消息
        """
        access_id = message.get("access_id")
        uri = message.get("uri")
        
        logger.debug(f"处理资源访问: {uri}")
        
        # 资源处理映射
        resource_handlers = {
            "collections": self._resource_collections,
            "collection": self._resource_collection,
            "stats": self._resource_stats
        }
        
        # 解析URI
        parts = uri.split("/")
        resource_type = parts[0] if parts else ""
        
        # 检查资源类型是否存在
        if resource_type not in resource_handlers:
            await self._send_error(websocket, "unknown_resource", f"未知的资源类型: {resource_type}", access_id=access_id)
            return
        
        try:
            # 调用对应的资源处理函数
            handler = resource_handlers[resource_type]
            result = await handler(uri, parts[1:] if len(parts) > 1 else [])
            
            # 发送成功响应
            await self._send_message(websocket, {
                "type": "resource_response",
                "access_id": access_id,
                "status": "success",
                "data": result
            })
        except Exception as e:
            # 发送错误响应
            logger.error(f"资源访问失败: {str(e)}")
            await self._send_error(websocket, "resource_access_error", str(e), access_id=access_id)
    
    async def _send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        发送消息到客户端
        
        Args:
            websocket: WebSocket连接
            message: 消息内容
        """
        await websocket.send_text(json.dumps(message))
    
    async def _send_error(self, websocket: WebSocket, error_type: str, error_message: str, call_id: str = None, access_id: str = None):
        """
        发送错误消息到客户端
        
        Args:
            websocket: WebSocket连接
            error_type: 错误类型
            error_message: 错误消息
            call_id: 工具调用ID（可选）
            access_id: 资源访问ID（可选）
        """
        message = {
            "type": "error",
            "error_type": error_type,
            "error_message": error_message
        }
        
        if call_id:
            message["call_id"] = call_id
        
        if access_id:
            message["access_id"] = access_id
        
        await self._send_message(websocket, message)
    
    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        获取工具模式定义
        
        Returns:
            工具模式列表
        """
        return [
            {
                "name": "create_collection",
                "description": "创建新的向量集合",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "集合名称"
                        },
                        "vector_size": {
                            "type": "integer",
                            "description": "向量维度大小",
                            "default": 768
                        },
                        "description": {
                            "type": "string",
                            "description": "集合描述",
                            "default": ""
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "list_collections",
                "description": "获取所有向量集合列表",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "返回结果数量限制",
                            "default": 100
                        },
                        "offset": {
                            "type": "integer",
                            "description": "结果偏移量",
                            "default": 0
                        }
                    }
                }
            },
            {
                "name": "delete_collection",
                "description": "删除向量集合",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "集合名称"
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "add_vectors",
                "description": "添加向量到集合",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "集合名称"
                        },
                        "texts": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "文本列表，将自动转换为向量"
                        },
                        "metadatas": {
                            "type": "array",
                            "items": {
                                "type": "object"
                            },
                            "description": "元数据列表，与文本一一对应"
                        }
                    },
                    "required": ["collection_name", "texts"]
                }
            },
            {
                "name": "search_by_text",
                "description": "通过文本搜索相似内容",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "集合名称"
                        },
                        "text": {
                            "type": "string",
                            "description": "查询文本"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 10
                        },
                        "filter": {
                            "type": "object",
                            "description": "过滤条件"
                        }
                    },
                    "required": ["collection_name", "text"]
                }
            },
            {
                "name": "search_by_vector",
                "description": "通过向量搜索相似内容",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "集合名称"
                        },
                        "vector": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            },
                            "description": "查询向量"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 10
                        },
                        "filter": {
                            "type": "object",
                            "description": "过滤条件"
                        }
                    },
                    "required": ["collection_name", "vector"]
                }
            }
        ]
    
    def _get_resources_schema(self) -> List[Dict[str, Any]]:
        """
        获取资源模式定义
        
        Returns:
            资源模式列表
        """
        return [
            {
                "name": "collections",
                "description": "获取所有向量集合信息",
                "uri_template": "collections"
            },
            {
                "name": "collection",
                "description": "获取特定向量集合信息",
                "uri_template": "collection/{name}"
            },
            {
                "name": "stats",
                "description": "获取系统统计信息",
                "uri_template": "stats"
            }
        ]
    
    # 工具处理函数
    
    async def _tool_create_collection(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建集合工具
        
        Args:
            arguments: 工具参数
            
        Returns:
            创建结果
        """
        from src.models.collection import CollectionCreate
        
        collection = CollectionCreate(
            name=arguments["name"],
            vector_size=arguments.get("vector_size", 768),
            description=arguments.get("description", "")
        )
        
        result = await self.vector_db_service.create_collection(collection)
        return result.dict()
    
    async def _tool_list_collections(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取集合列表工具
        
        Args:
            arguments: 工具参数
            
        Returns:
            集合列表
        """
        limit = arguments.get("limit", 100)
        offset = arguments.get("offset", 0)
        
        result = await self.vector_db_service.list_collections(limit, offset)
        return result.dict()
    
    async def _tool_delete_collection(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        删除集合工具
        
        Args:
            arguments: 工具参数
            
        Returns:
            删除结果
        """
        name = arguments["name"]
        
        result = await self.vector_db_service.delete_collection(name)
        return result.dict()
    
    async def _tool_add_vectors(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加向量工具
        
        Args:
            arguments: 工具参数
            
        Returns:
            添加结果
        """
        from src.models.vector import VectorUpload, VectorBatchUpload
        
        collection_name = arguments["collection_name"]
        texts = arguments["texts"]
        metadatas = arguments.get("metadatas", [{}] * len(texts))
        
        # 确保metadatas长度与texts一致
        if len(metadatas) < len(texts):
            metadatas.extend([{}] * (len(texts) - len(metadatas)))
        
        # 获取文本嵌入
        embeddings = await self.embedding_service.get_embeddings(texts)
        
        # 准备向量数据
        vectors = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # 添加文本到元数据
            metadata["text"] = text
            
            vector = VectorUpload(
                vector=embedding,
                payload=metadata
            )
            vectors.append(vector)
        
        # 批量添加向量
        batch = VectorBatchUpload(vectors=vectors)
        result = await self.vector_db_service.add_vectors(collection_name, batch)
        
        return result.dict()
    
    async def _tool_search_by_text(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        通过文本搜索工具
        
        Args:
            arguments: 工具参数
            
        Returns:
            搜索结果
        """
        collection_name = arguments["collection_name"]
        text = arguments["text"]
        limit = arguments.get("limit", 10)
        filter_dict = arguments.get("filter")
        
        # 获取文本嵌入
        query_vector = await self.embedding_service.get_embeddings(text)
        
        # 执行向量搜索
        results = await self.vector_db_service.search_by_vector(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            filter=filter_dict
        )
        
        # 转换结果
        return [result.dict() for result in results]
    
    async def _tool_search_by_vector(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        通过向量搜索工具
        
        Args:
            arguments: 工具参数
            
        Returns:
            搜索结果
        """
        collection_name = arguments["collection_name"]
        query_vector = arguments["vector"]
        limit = arguments.get("limit", 10)
        filter_dict = arguments.get("filter")
        
        # 执行向量搜索
        results = await self.vector_db_service.search_by_vector(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            filter=filter_dict
        )
        
        # 转换结果
        return [result.dict() for result in results]
    
    # 资源处理函数
    
    async def _resource_collections(self, uri: str, path_parts: List[str]) -> Dict[str, Any]:
        """
        获取集合列表资源
        
        Args:
            uri: 资源URI
            path_parts: URI路径部分
            
        Returns:
            集合列表
        """
        result = await self.vector_db_service.list_collections()
        return result.dict()
    
    async def _resource_collection(self, uri: str, path_parts: List[str]) -> Dict[str, Any]:
        """
        获取集合信息资源
        
        Args:
            uri: 资源URI
            path_parts: URI路径部分
            
        Returns:
            集合信息
        """
        if not path_parts:
            raise ValueError("缺少集合名称")
        
        name = path_parts[0]
        result = await self.vector_db_service.get_collection(name)
        return result.dict()
    
    async def _resource_stats(self, uri: str, path_parts: List[str]) -> Dict[str, Any]:
        """
        获取统计信息资源
        
        Args:
            uri: 资源URI
            path_parts: URI路径部分
            
        Returns:
            统计信息
        """
        # 获取集合列表
        collections = await self.vector_db_service.list_collections()
        
        # 计算总向量数
        total_vectors = sum(collection.vector_count for collection in collections.collections)
        
        return {
            "collections_count": len(collections.collections),
            "total_vectors": total_vectors,
            "embedding_model": self.embedding_service.model_name,
            "vector_dimension": self.embedding_service.vector_size
        }