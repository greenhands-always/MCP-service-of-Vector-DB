"""
RAG（检索增强生成）应用示例
"""
import asyncio
import os
import sys
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db.vector_db_service import VectorDBService
from src.core.embedding import EmbeddingService
from src.models.collection import CollectionCreate
from src.models.vector import VectorUpload, VectorBatchUpload
from src.utils.text_splitter import ChineseTextSplitter

# 模拟LLM接口（实际应用中应替换为真实的LLM API调用）
async def mock_llm_generate(prompt: str, context: List[str]) -> str:
    """
    模拟LLM生成回答
    
    Args:
        prompt: 用户提问
        context: 检索到的上下文
        
    Returns:
        生成的回答
    """
    # 在实际应用中，这里应该调用真实的LLM API
    return f"""基于检索到的信息，我可以回答您的问题：

向量数据库在RAG架构中扮演着关键角色，主要有以下几个方面的作用：

1. 存储知识库：将文档、网页、知识库等转换为向量嵌入并存储
2. 语义检索：根据用户查询找到最相关的信息片段
3. 上下文增强：为LLM提供相关上下文，使其生成更准确的回答

这种方式使得大语言模型能够基于检索到的最新、最相关的信息生成回答，而不仅仅依赖于其预训练知识。"""

async def rag_pipeline(query: str, collection_name: str, vector_db: VectorDBService, embedding_service: EmbeddingService) -> str:
    """
    RAG流程实现
    
    Args:
        query: 用户查询
        collection_name: 集合名称
        vector_db: 向量数据库服务
        embedding_service: 嵌入服务
        
    Returns:
        生成的回答
    """
    # 1. 将查询转换为向量
    print(f"将查询转换为向量: {query}")
    query_vector = await embedding_service.get_embeddings(query)
    
    # 2. 在向量数据库中搜索相似内容
    print("在向量数据库中搜索相似内容...")
    search_results = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    
    # 3. 提取检索到的上下文
    contexts = []
    for result in search_results:
        contexts.append(result.payload["text"])
        print(f"检索到相关内容 (相似度: {result.score:.4f}): {result.payload['text'][:100]}...")
    
    # 4. 构建提示并调用LLM生成回答
    print("生成回答...")
    answer = await mock_llm_generate(query, contexts)
    
    return answer

async def main():
    """
    主函数
    """
    print("RAG应用示例")
    
    # 初始化服务
    vector_db = VectorDBService()
    embedding_service = EmbeddingService()
    
    # 创建测试集合
    collection_name = "rag_demo"
    vector_size = embedding_service.vector_size
    
    # 检查集合是否已存在，如果存在则删除
    collection = await vector_db.get_collection(collection_name)
    if collection:
        print(f"删除已存在的集合: {collection_name}")
        await vector_db.delete_collection(collection_name)
    
    # 创建新集合
    print(f"创建新集合: {collection_name}")
    collection_create = CollectionCreate(
        name=collection_name,
        vector_size=vector_size,
        description="RAG演示集合"
    )
    collection = await vector_db.create_collection(collection_create)
    print(f"集合创建成功: {collection.name}, 向量维度: {collection.vector_size}")
    
    # 准备知识库文档
    knowledge_base = """
# 向量数据库在RAG中的应用

检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合了检索系统和生成式AI的技术架构，用于提高大语言模型（LLM）生成内容的准确性、相关性和可靠性。

在RAG架构中，向量数据库扮演着关键角色：

1. 存储知识库：将文档、网页、知识库等转换为向量嵌入并存储
2. 语义检索：根据用户查询找到最相关的信息片段
3. 上下文增强：为LLM提供相关上下文，使其生成更准确的回答

RAG的典型工作流程如下：

1. 索引阶段：
   - 收集和预处理文档
   - 将文档分割成适当大小的块
   - 使用嵌入模型将文本块转换为向量嵌入
   - 将向量嵌入和原始文本存储在向量数据库中

2. 检索阶段：
   - 接收用户查询
   - 将查询转换为向量嵌入
   - 在向量数据库中搜索与查询向量最相似的文档向量
   - 检索相应的原始文本片段

3. 生成阶段：
   - 将检索到的文本片段作为上下文提供给LLM
   - LLM基于查询和提供的上下文生成回答

向量数据库的性能和功能直接影响RAG系统的检索质量和响应速度，因此选择合适的向量数据库解决方案对构建高效的RAG应用至关重要。

## 向量数据库优化方向

除了选择合适的向量数据库外，优化RAG系统还可以从以下方面入手：

1. 文档处理优化：
   - 更智能的文档分块策略
   - 重叠分块以保留上下文
   - 文档预处理和清洗

2. 检索策略优化：
   - 混合检索（关键词+语义）
   - 多查询检索
   - 查询重写和扩展

3. 排序和重排序：
   - 检索后重排序
   - 相关性评分优化
   - 多阶段检索

4. 嵌入模型选择：
   - 领域特定嵌入模型
   - 微调通用嵌入模型
   - 多模态嵌入

5. 系统架构优化：
   - 缓存机制
   - 预计算和批处理
   - 异步处理
"""
    
    # 使用文本分割器处理文档
    print("分割文档...")
    text_splitter = ChineseTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(knowledge_base)
    print(f"文档被分割为 {len(chunks)} 个块")
    
    # 生成文档嵌入
    print("生成文档嵌入...")
    embeddings = await embedding_service.get_embeddings(chunks)
    
    # 批量添加向量
    print("将文档块添加到向量数据库...")
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append(
            VectorUpload(
                id=f"chunk_{i}",
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": "RAG知识库",
                    "index": i
                }
            )
        )
    
    batch_upload = VectorBatchUpload(vectors=vectors)
    result = await vector_db.add_vectors(collection_name, batch_upload)
    print(f"添加了 {result.added_count} 个向量")
    
    # 执行RAG查询
    print("\n执行RAG查询示例:")
    
    # 用户查询
    query = "向量数据库在RAG架构中的作用是什么？"
    print(f"用户查询: {query}")
    
    # 执行RAG流程
    answer = await rag_pipeline(query, collection_name, vector_db, embedding_service)
    
    print("\n生成的回答:")
    print(answer)
    
    # 清理
    print("\n清理测试数据...")
    await vector_db.delete_collection(collection_name)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(main())