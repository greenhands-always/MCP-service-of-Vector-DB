# 向量数据库MCP服务

一个轻量级的向量数据库服务，将Qdrant向量数据库转换为MCP（Model Context Protocol）服务，用于支持RAG（检索增强生成）应用开发。

## 项目概述

向量数据库MCP服务是一个专门设计用于存储、索引和查询向量嵌入的服务系统，并通过MCP协议提供服务。它为RAG应用提供了核心的检索功能，使得大语言模型能够基于相关上下文生成更准确、更可靠的回答。

### 主要功能

- 文档向量化和存储
- 高效的相似性搜索
- 支持元数据过滤
- 多集合（collection）管理
- MCP协议接口
- RESTful API接口

## 快速开始

### 环境要求

- Python 3.8+
- Docker (推荐)

### 使用Docker运行

最简单的方式是使用Docker Compose启动服务：

```bash
# 克隆仓库
git clone https://github.com/your-org/vector-db-service.git
cd vector-db-service

# 启动服务
docker-compose up -d
```

这将启动两个服务：
1. Qdrant向量数据库（端口6333）
2. 向量数据库MCP服务（端口8000）

### 手动安装

如果您不想使用Docker，也可以手动安装：

```bash
# 克隆仓库
git clone https://github.com/your-org/vector-db-service.git
cd vector-db-service

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .

# 启动Qdrant（需要单独安装）
# 参考：https://qdrant.tech/documentation/install/

# 配置环境变量
export VECTOR_DB_HOST=localhost  # 或设置到.env文件中
export VECTOR_DB_PORT=8000
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# 启动服务
python -m src.main
```

## 使用MCP服务

### MCP协议

MCP（Model Context Protocol）是一种用于模型上下文交互的协议。本服务通过WebSocket提供MCP接口，端点为：

```
ws://localhost:8000/mcp
```

### MCP客户端示例

```python
import asyncio
import websockets
import json
import uuid

async def main():
    # 连接到MCP服务
    async with websockets.connect("ws://localhost:8000/mcp") as websocket:
        # 接收服务器信息
        response = await websocket.recv()
        server_info = json.loads(response)
        print(f"已连接到MCP服务: {server_info['result']['name']}")
        
        # 创建集合
        request_id = str(uuid.uuid4())
        await websocket.send(json.dumps({
            "id": request_id,
            "method": "create_collection",
            "params": {
                "name": "test_collection",
                "description": "测试集合"
            }
        }))
        
        # 接收响应
        response = await websocket.recv()
        print(f"创建集合响应: {response}")
        
        # 添加向量
        request_id = str(uuid.uuid4())
        await websocket.send(json.dumps({
            "id": request_id,
            "method": "add_vector",
            "params": {
                "collection_name": "test_collection",
                "text": "这是一个测试文档",
                "payload": {"source": "测试"}
            }
        }))
        
        # 接收响应
        response = await websocket.recv()
        print(f"添加向量响应: {response}")
        
        # 搜索向量
        request_id = str(uuid.uuid4())
        await websocket.send(json.dumps({
            "id": request_id,
            "method": "search_vectors",
            "params": {
                "collection_name": "test_collection",
                "query_text": "测试查询",
                "limit": 5
            }
        }))
        
        # 接收响应
        response = await websocket.recv()
        print(f"搜索响应: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

更完整的示例请参见 `examples/mcp_client_example.py`。

## 也可以使用REST API

除了MCP接口外，本服务也提供了RESTful API接口：

```python
import requests

# 创建集合
response = requests.post(
    "http://localhost:8000/api/v1/collections",
    json={"name": "documents", "vector_size": 768}
)

# 添加向量
response = requests.post(
    "http://localhost:8000/api/v1/vectors",
    params={"collection_name": "documents"},
    json={
        "id": "1",
        "vector": [0.1, 0.2, ...],  # 向量数据
        "payload": {"text": "这是一个示例文档", "source": "示例"}
    }
)

# 搜索向量
response = requests.post(
    "http://localhost:8000/api/v1/search/text",
    params={"collection_name": "documents"},
    json={
        "text": "查询文本",
        "limit": 5,
        "filter": {"source": "示例"}
    }
)
```

## 项目结构

```
vector-db-service/
├── src/                  # 源代码
│   ├── api/              # REST API接口定义
│   ├── core/             # 核心业务逻辑
│   ├── db/               # 数据库交互
│   ├── mcp/              # MCP服务实现
│   ├── models/           # 数据模型
│   └── utils/            # 工具函数
├── examples/             # 使用示例
├── PRDs/                 # 需求文档
└── docker-compose.yml    # Docker配置
```

## MCP工具列表

本服务提供以下MCP工具：

1. `search_vectors`: 搜索向量
2. `add_vector`: 添加单个向量
3. `add_vectors`: 批量添加向量
4. `get_embedding`: 获取文本嵌入向量
5. `list_collections`: 获取集合列表
6. `create_collection`: 创建新集合

## 许可证

本项目采用[MIT许可证](LICENSE)。
