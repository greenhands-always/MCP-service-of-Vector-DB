# BERT向量化使用指南

本文档详细介绍如何在向量数据库MCP服务中使用BERT模型进行文本向量化，以提高语义搜索的质量和相关性。

## 目录

1. [简介](#简介)
2. [安装依赖](#安装依赖)
3. [配置BERT模型](#配置bert模型)
4. [推荐的BERT模型](#推荐的bert模型)
5. [性能考虑](#性能考虑)
6. [使用示例](#使用示例)
7. [高级用法](#高级用法)
8. [故障排除](#故障排除)

## 简介

BERT（Bidirectional Encoder Representations from Transformers）是一种预训练的语言模型，能够生成上下文感知的文本表示。在RAG（检索增强生成）应用中，使用BERT进行文本向量化可以显著提高检索的准确性和相关性。

本项目通过集成`sentence-transformers`库，支持使用各种BERT模型变体进行文本向量化。

## 安装依赖

要使用BERT向量化功能，首先需要安装必要的依赖：

```bash
# 使用pip安装
pip install ".[embeddings]"

# 或者使用uv安装
uv pip install ".[embeddings]"
```

这将安装`sentence-transformers`库及其依赖项，包括`torch`和`transformers`。

## 配置BERT模型

配置BERT模型有两种方式：

### 1. 通过环境变量配置

创建或修改`.env`文件，添加以下配置：

```
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=paraphrase-MiniLM-L6-v2  # 或其他BERT模型
EMBEDDING_DEVICE=cpu  # 如果有GPU，可以设置为cuda
```

### 2. 使用预配置的环境文件

我们提供了一个预配置的环境文件`.env.bert`，您可以直接使用：

```bash
cp .env.bert .env
```

然后根据需要修改`.env`文件中的模型配置。

## 推荐的BERT模型

### 中文模型

| 模型名称 | 向量维度 | 特点 | 适用场景 |
|---------|---------|------|---------|
| `BAAI/bge-small-zh` | 384 | 轻量级中文模型，速度快 | 资源受限场景，一般中文应用 |
| `BAAI/bge-large-zh` | 1024 | 大型中文模型，效果好 | 对精度要求高的中文应用 |
| `shibing624/text2vec-base-chinese` | 768 | 专为中文语义相似度优化 | 中文语义搜索应用 |

### 英文/多语言模型

| 模型名称 | 向量维度 | 特点 | 适用场景 |
|---------|---------|------|---------|
| `paraphrase-MiniLM-L6-v2` | 384 | 轻量级模型，平衡性能和效率 | 一般英文应用，资源受限场景 |
| `all-MiniLM-L6-v2` | 384 | 支持多语言的轻量级模型 | 多语言应用 |
| `all-mpnet-base-v2` | 768 | 效果更好但更大的模型 | 对精度要求高的应用 |
| `distilbert-base-nli-stsb-mean-tokens` | 768 | 平衡大小和性能的模型 | 一般英文应用 |

## 性能考虑

使用BERT模型进行向量化时，需要考虑以下性能因素：

### 1. 模型加载时间

- 首次加载BERT模型时会下载模型文件，可能需要一些时间
- 模型大小影响加载时间，小型模型（如`MiniLM`系列）加载更快
- 模型文件会缓存到本地，后续加载会更快

### 2. 向量化速度

- 在CPU上运行可能较慢，特别是处理大量文本时
- 如果有GPU，设置`EMBEDDING_DEVICE=cuda`可显著提高速度
- 批处理多个文本比单独处理每个文本更高效

### 3. 内存使用

- BERT模型需要较大内存，根据模型大小可能需要2GB-5GB内存
- 处理大量文本时，注意批处理大小，避免内存溢出

### 4. 向量维度

- 不同模型生成的向量维度不同（384-1024维）
- 向量维度影响存储空间和检索速度
- 创建集合时需要指定正确的向量维度

## 使用示例

我们提供了多个示例脚本，展示如何使用BERT进行向量化：

### 1. 基本向量化示例

```bash
python examples/bert_embedding_example.py
```

这个示例展示了如何使用BERT模型生成文本嵌入，并计算文本之间的相似度。

### 2. RAG应用示例

```bash
python examples/bert_rag_example.py
```

这个示例展示了如何在RAG应用中使用BERT模型进行文档检索。

### 3. 批量文档处理示例

```bash
python examples/bert_batch_processing.py
```

这个示例展示了如何高效处理大量文档，包括文本分块、批量向量化和过滤搜索。

### 4. 模型评估示例

```bash
python examples/bert_model_evaluation.py
```

这个示例比较不同BERT模型的性能和效果，帮助选择最适合特定应用的模型。

## 高级用法

### 1. 自定义文本分割

对于长文档，合理的文本分割策略对检索效果至关重要：

```python
from src.utils.text_splitter import ChineseTextSplitter

# 自定义分割参数
text_splitter = ChineseTextSplitter(
    chunk_size=300,     # 每个块的目标大小（字符数）
    chunk_overlap=50,   # 块之间的重叠大小（字符数）
    separator="\n"      # 优先在换行处分割
)

chunks = text_splitter.split_text(long_document)
```

### 2. 带元数据的向量化

向量可以附加元数据，便于后续过滤和组织：

```python
from src.core.embedding import EmbeddingService

embedding_service = EmbeddingService()

texts = ["文本1", "文本2", "文本3"]
metadatas = [
    {"source": "文档1", "category": "技术"},
    {"source": "文档2", "category": "新闻"},
    {"source": "文档3", "category": "教育"}
]

result = await embedding_service.get_embeddings_with_metadata(texts, metadatas)
```

### 3. 过滤搜索

结合向量相似度和元数据过滤进行更精确的搜索：

```python
from src.db.vector_db_service import VectorDBService

vector_db = VectorDBService()

# 按类别过滤
results = await vector_db.search_by_vector(
    collection_name="my_collection",
    query_vector=query_vector,
    limit=5,
    filter={"metadata.category": "技术"}
)

# 按多个条件过滤
results = await vector_db.search_by_vector(
    collection_name="my_collection",
    query_vector=query_vector,
    limit=5,
    filter={
        "metadata.category": "技术",
        "metadata.source": ["文档1", "文档2"]
    }
)
```

## 故障排除

### 1. 模型加载失败

如果遇到模型加载失败的问题：

- 检查网络连接，确保可以访问Hugging Face模型仓库
- 检查磁盘空间，确保有足够空间下载和缓存模型
- 尝试使用较小的模型，如`paraphrase-MiniLM-L6-v2`

### 2. 内存错误

如果遇到内存不足错误：

- 减小批处理大小
- 使用较小的模型
- 增加系统内存或使用虚拟内存

### 3. GPU相关问题

如果使用GPU但遇到问题：

- 确保已正确安装CUDA和cuDNN
- 检查GPU内存是否足够
- 如果问题持续，尝试切换到CPU模式：`EMBEDDING_DEVICE=cpu`

### 4. 向量维度不匹配

如果遇到向量维度不匹配错误：

- 确保创建集合时指定的向量维度与模型输出维度一致
- 使用`embedding_service.vector_size`获取当前模型的向量维度
- 如果更换了模型，可能需要重新创建集合

---

如有更多问题，请参考项目文档或提交Issue。