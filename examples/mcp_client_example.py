"""
MCP客户端示例，演示如何连接和使用向量数据库MCP服务
"""
import asyncio
import json
import sys
import os
import uuid
import websockets
from typing import Dict, Any, List, Optional

class MCPClient:
    """
    MCP客户端类
    """
    
    def __init__(self, url: str = "ws://localhost:8000/mcp"):
        """
        初始化MCP客户端
        
        Args:
            url: MCP服务器WebSocket URL
        """
        self.url = url
        self.websocket = None
        self.server_info = None
    
    async def connect(self):
        """
        连接到MCP服务
        """
        self.websocket = await websockets.connect(self.url)
        
        # 接收服务器信息
        response = await self.websocket.recv()
        self.server_info = json.loads(response)
        
        print(f"已连接到MCP服务: {self.server_info['result']['name']} v{self.server_info['result']['version']}")
        print(f"服务描述: {self.server_info['result']['description']}")
        print("可用工具:")
        for tool in self.server_info['result']['tools']:
            print(f"  - {tool['name']}: {tool['description']}")
    
    async def close(self):
        """
        关闭连接
        """
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用MCP方法
        
        Args:
            method: 方法名
            params: 参数
            
        Returns:
            响应结果
        """
        if not self.websocket:
            raise RuntimeError("未连接到MCP服务")
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 构建请求
        request = {
            "id": request_id,
            "method": method,
            "params": params
        }
        
        # 发送请求
        await self.websocket.send(json.dumps(request))
        
        # 接收响应
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        # 检查错误
        if "error" in response_data and response_data["error"]:
            raise Exception(f"MCP错误: {response_data['error']['message']}")
        
        return response_data["result"]

async def demo_vector_search():
    """
    演示向量搜索功能
    """
    client = MCPClient()
    
    try:
        # 连接到MCP服务
        await client.connect()
        
        # 创建测试集合
        collection_name = "mcp_demo"
        print(f"\n创建集合: {collection_name}")
        try:
            await client.call("create_collection", {
                "name": collection_name,
                "description": "MCP演示集合"
            })
            print("集合创建成功")
        except Exception as e:
            print(f"创建集合失败（可能已存在）: {str(e)}")
        
        # 添加测试数据
        print("\n添加测试数据...")
        documents = [
            "向量数据库是一种专门设计用于存储、索引和查询向量嵌入的数据库系统。",
            "向量嵌入是将文本、图像、音频等非结构化数据转换为高维数字向量的表示方法。",
            "向量数据库的核心功能是高效地执行相似性搜索，如最近邻搜索（k-NN）或近似最近邻搜索（ANN）。",
            "RAG（检索增强生成）是一种结合了检索系统和生成式AI的技术架构。",
            "在RAG架构中，向量数据库扮演着存储知识库、语义检索和上下文增强的关键角色。"
        ]
        
        # 批量添加向量
        vectors = []
        for i, text in enumerate(documents):
            vectors.append({
                "id": f"doc_{i}",
                "text": text,
                "payload": {
                    "source": "MCP演示",
                    "index": i
                }
            })
        
        result = await client.call("add_vectors", {
            "collection_name": collection_name,
            "vectors": vectors
        })
        
        print(f"添加了 {result['added_count']} 个向量")
        
        # 执行搜索
        print("\n执行搜索...")
        query = "RAG架构中向量数据库的作用是什么？"
        print(f"查询: {query}")
        
        search_result = await client.call("search_vectors", {
            "collection_name": collection_name,
            "query_text": query,
            "limit": 3
        })
        
        print("\n搜索结果:")
        for i, result in enumerate(search_result["results"]):
            print(f"{i+1}. 相似度: {result['score']:.4f}")
            print(f"   文本: {result['payload'].get('text', '')}")
            print()
        
        # 获取嵌入向量示例
        print("\n获取嵌入向量示例:")
        text = "这是一个测试文本"
        embedding = await client.call("get_embedding", {"text": text})
        print(f"文本 '{text}' 的嵌入向量维度: {embedding['dimension']}")
        print(f"向量前5个元素: {embedding['vector'][:5]}...")
        
        # 列出所有集合
        print("\n列出所有集合:")
        collections = await client.call("list_collections", {})
        for collection in collections["collections"]:
            print(f"- {collection['name']}: {collection['vector_count']} 个向量, 维度: {collection['vector_size']}")
    
    finally:
        # 关闭连接
        await client.close()

if __name__ == "__main__":
    asyncio.run(demo_vector_search())