"""
使用BERT模型的RAG（检索增强生成）应用示例
"""
import asyncio
import os
import sys
import time
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db.vector_db_service import VectorDBService
from src.core.embedding import EmbeddingService
from src.models.collection import CollectionCreate
from src.models.vector import VectorUpload, VectorBatchUpload
from src.utils.text_splitter import ChineseTextSplitter
from src.core.config import settings

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
    context_text = "\n\n".join(context)
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
    start_time = time.time()
    query_vector = await embedding_service.get_embeddings(query)
    query_time = time.time() - start_time
    print(f"查询向量生成完成，耗时: {query_time:.2f}秒")
    
    # 2. 在向量数据库中搜索相似内容
    print("在向量数据库中搜索相似内容...")
    start_time = time.time()
    search_results = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    search_time = time.time() - start_time
    print(f"搜索完成，耗时: {search_time:.2f}秒")
    
    # 3. 提取检索到的上下文
    contexts = []
    for result in search_results:
        contexts.append(result.payload["text"])
        print(f"检索到相关内容 (相似度: {result.score:.4f}): {result.payload['text'][:100]}...")
    
    # 4. 构建提示并调用LLM生成回答
    print("生成回答...")
    start_time = time.time()
    answer = await mock_llm_generate(query, contexts)
    generate_time = time.time() - start_time
    print(f"回答生成完成，耗时: {generate_time:.2f}秒")
    
    # 5. 计算总耗时
    total_time = query_time + search_time + generate_time
    print(f"RAG流程总耗时: {total_time:.2f}秒")
    
    return answer

async def main():
    """
    主函数
    """
    print("使用BERT模型的RAG应用示例")
    print(f"当前配置的嵌入提供者: {settings.EMBEDDING_PROVIDER}")
    print(f"当前配置的嵌入模型: {settings.EMBEDDING_MODEL}")
    print(f"当前配置的设备: {settings.EMBEDDING_DEVICE}")
    
    # 初始化服务
    print("\n初始化服务...")
    start_time = time.time()
    vector_db = VectorDBService()
    embedding_service = EmbeddingService()
    init_time = time.time() - start_time
    print(f"服务初始化完成，耗时: {init_time:.2f}秒")
    print(f"向量维度: {embedding_service.vector_size}")
    
    # 创建测试集合
    collection_name = "bert_rag_demo"
    vector_size = embedding_service.vector_size
    
    # 检查集合是否已存在，如果存在则删除
    try:
        collection = await vector_db.get_collection(collection_name)
        if collection:
            print(f"删除已存在的集合: {collection_name}")
            await vector_db.delete_collection(collection_name)
    except Exception:
        pass
    
    # 创建新集合
    print(f"创建新集合: {collection_name}")
    collection_create = CollectionCreate(
        name=collection_name,
        vector_size=vector_size,
        description="BERT RAG演示集合"
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

## BERT在RAG中的应用

BERT（Bidirectional Encoder Representations from Transformers）是一种预训练的语言模型，在RAG系统中主要用于生成文本的向量表示。

BERT的优势：

1. 双向上下文理解：BERT能够同时考虑文本的左右上下文，捕获更丰富的语义信息
2. 预训练-微调范式：可以在通用语料上预训练，然后在特定领域微调
3. 多语言支持：存在多种语言版本，如中文BERT、多语言BERT等
4. 丰富的变体：如RoBERTa、DistilBERT等，提供不同的性能和效率权衡

在RAG系统中，BERT主要用于：

1. 文档嵌入：将知识库文档转换为向量表示
2. 查询嵌入：将用户查询转换为向量表示
3. 语义匹配：通过向量相似度计算实现查询与文档的语义匹配

针对中文RAG应用，推荐使用以下BERT变体：

1. BAAI/bge-small-zh：轻量级中文BERT模型，适合资源受限场景
2. BAAI/bge-large-zh：大型中文BERT模型，提供更高的语义理解能力
3. GanymedeNil/text2vec-large-chinese：专为中文语义相似度优化的模型

使用BERT进行向量化时的最佳实践：

1. 选择适合领域的预训练模型
2. 考虑模型大小与性能的平衡
3. 使用批处理提高效率
4. 在可能的情况下使用GPU加速
5. 考虑向量压缩技术减少存储需求
"""
    
    # 使用文本分割器处理文档
    print("分割文档...")
    text_splitter = ChineseTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(knowledge_base)
    print(f"文档被分割为 {len(chunks)} 个块")
    
    # 生成文档嵌入
    print("生成文档嵌入...")
    start_time = time.time()
    embeddings = await embedding_service.get_embeddings(chunks)
    embed_time = time.time() - start_time
    print(f"文档嵌入生成完成，耗时: {embed_time:.2f}秒")
    
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
                    "source": "BERT RAG知识库",
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
    queries = [
        "向量数据库在RAG架构中的作用是什么？",
        "BERT模型在RAG系统中有什么优势？",
        "如何优化RAG系统的检索质量？"
    ]
    
    # 执行多个查询
    for i, query in enumerate(queries):
        print(f"\n查询 {i+1}: {query}")
        
        # 执行RAG流程
        answer = await rag_pipeline(query, collection_name, vector_db, embedding_service)
        
        print("\n生成的回答:")
        print(answer)
        print("-" * 80)
    
    # 清理
    print("\n清理测试数据...")
    await vector_db.delete_collection(collection_name)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(main())