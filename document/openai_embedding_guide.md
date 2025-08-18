# OpenAI兼容协议嵌入服务使用指南

本文档介绍如何在向量数据库服务中使用OpenAI兼容协议的嵌入服务。

## 概述

OpenAI兼容协议嵌入服务允许您通过API调用的方式获取文本的向量表示，支持：

- OpenAI官方API
- 其他兼容OpenAI协议的服务（如DeepSeek、智谱AI等）
- 本地部署的兼容服务

## 配置方式

### 1. 环境变量配置

在`.env`文件中添加以下配置：

```bash
# 设置嵌入服务提供商为OpenAI
EMBEDDING_PROVIDER=openai

# OpenAI API配置
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### 2. 支持的服务提供商

#### OpenAI官方API
```bash
OPENAI_API_KEY=sk-your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

#### DeepSeek API
```bash
OPENAI_API_KEY=your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-v1
```

#### 智谱AI API
```bash
OPENAI_API_KEY=your-zhipu-key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_EMBEDDING_MODEL=embedding-2
```

#### 本地部署服务
```bash
OPENAI_API_KEY=local-key
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_EMBEDDING_MODEL=bge-large-zh-v1.5
```

## 使用方式

### 1. 直接创建实例

```python
from src.core.embedding import OpenAIEmbedding

# 创建嵌入服务实例
embedding_service = OpenAIEmbedding(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model_name="text-embedding-ada-002",
    timeout=30,
    max_retries=3
)

# 获取单个文本的嵌入向量
text = "这是一个测试文本"
embedding = await embedding_service.get_embeddings(text)
print(f"嵌入维度: {len(embedding)}")

# 获取多个文本的嵌入向量
texts = ["文本1", "文本2", "文本3"]
embeddings = await embedding_service.get_embeddings(texts)
print(f"批量嵌入数量: {len(embeddings)}")
```

### 2. 使用工厂函数

```python
from src.core.embedding import create_embedding_service

# 根据配置创建嵌入服务
embedding_service = create_embedding_service('openai')

# 使用服务
embedding = await embedding_service.get_embeddings("测试文本")
```

### 3. 带元数据的嵌入

```python
texts = ["文本1", "文本2"]
metadatas = [
    {"category": "技术", "source": "文档"},
    {"category": "产品", "source": "说明书"}
]

result = await embedding_service.get_embeddings_with_metadata(texts, metadatas)
print(f"结果: {result}")
```

## API接口规范

OpenAI兼容协议的嵌入服务遵循以下API规范：

### 请求格式

```http
POST /embeddings
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "model": "text-embedding-ada-002",
    "input": ["文本1", "文本2"]
}
```

### 响应格式

```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3, ...]
        },
        {
            "object": "embedding", 
            "index": 1,
            "embedding": [0.4, 0.5, 0.6, ...]
        }
    ],
    "model": "text-embedding-ada-002",
    "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10
    }
}
```

## 错误处理

服务包含完善的错误处理机制：

- **重试机制**: 自动重试失败的请求，默认最多重试3次
- **超时处理**: 支持设置请求超时时间，默认30秒
- **错误日志**: 详细记录错误信息，便于调试

```python
try:
    embedding = await embedding_service.get_embeddings("测试文本")
except Exception as e:
    logger.error(f"获取嵌入向量失败: {str(e)}")
```

## 性能优化建议

1. **批量处理**: 尽量使用批量接口处理多个文本，减少API调用次数
2. **缓存机制**: 对于重复的文本，可以实现缓存机制避免重复计算
3. **并发控制**: 控制并发请求数量，避免超出API限制
4. **模型选择**: 根据需求选择合适的模型，平衡性能和成本

## 常见问题

### Q: 如何选择合适的嵌入模型？

A: 根据您的需求选择：
- `text-embedding-ada-002`: OpenAI的通用模型，性能均衡
- `text-embedding-3-small`: 更新的小型模型，速度快
- `text-embedding-3-large`: 更新的大型模型，精度高

### Q: 如何处理API限制？

A: 
- 实现请求频率限制
- 使用批量接口减少请求次数
- 监控API使用量

### Q: 本地部署的服务如何配置？

A: 确保本地服务实现了OpenAI兼容的API接口，然后设置正确的`OPENAI_BASE_URL`。

## 示例代码

完整的使用示例请参考：
- `examples/openai_embedding_example.py` - 基础使用示例
- `.env.openai` - 配置文件示例

## 相关文档

- [配置文档](config.md)
- [API文档](api.md)
- [部署指南](deployment.md)