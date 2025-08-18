"""
BERT批量文档处理示例 - 展示如何高效处理大量文档
"""
import asyncio
import os
import sys
import time
from typing import List, Dict, Any
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.embedding import EmbeddingService
from src.db.vector_db_service import VectorDBService
from src.models.collection import CollectionCreate
from src.models.vector import VectorUpload, VectorBatchUpload
from src.utils.text_splitter import ChineseTextSplitter
from src.core.config import settings

# 模拟文档数据
SAMPLE_DOCUMENTS = [
    {
        "title": "向量数据库简介",
        "content": """
        向量数据库是一种专门设计用于存储、索引和查询向量嵌入的数据库系统。与传统数据库不同，向量数据库专注于高效处理高维向量数据，
        支持相似性搜索操作，如最近邻搜索（k-NN）或近似最近邻搜索（ANN）。
        
        向量数据库的主要特点包括：
        1. 高效的相似性搜索
        2. 支持高维向量数据
        3. 可扩展性强
        4. 针对向量操作优化
        
        常见的向量数据库包括Qdrant、Milvus、Pinecone、Weaviate等。
        """,
        "metadata": {
            "source": "技术文档",
            "author": "数据库专家",
            "category": "数据库",
            "tags": ["向量数据库", "相似性搜索", "高维数据"]
        }
    },
    {
        "title": "BERT模型详解",
        "content": """
        BERT（Bidirectional Encoder Representations from Transformers）是由Google AI团队在2018年提出的预训练语言模型。
        它通过在大规模无标注文本上进行预训练，学习通用的语言表示，然后在特定任务上进行微调，实现了多项自然语言处理任务的突破。
        
        BERT的主要创新点在于：
        1. 双向上下文理解：同时考虑文本的左右上下文
        2. 预训练-微调范式：通用预训练+特定任务微调
        3. 掩码语言模型（MLM）：随机掩盖输入中的一些词，并预测这些被掩盖的词
        4. 下一句预测（NSP）：预测两个句子是否为连续的句子
        
        BERT已经成为NLP领域的基础模型，衍生出多种变体，如RoBERTa、DistilBERT、ALBERT等。
        """,
        "metadata": {
            "source": "AI研究论文",
            "author": "NLP研究员",
            "category": "自然语言处理",
            "tags": ["BERT", "预训练语言模型", "Transformer"]
        }
    },
    {
        "title": "RAG技术架构",
        "content": """
        RAG（检索增强生成）是一种结合了检索系统和生成式AI的技术架构，用于提高大语言模型（LLM）生成内容的准确性、相关性和可靠性。
        
        RAG的工作流程通常包括以下步骤：
        1. 索引阶段：将知识库文档转换为向量嵌入并存储在向量数据库中
        2. 检索阶段：将用户查询转换为向量，在向量数据库中检索相似内容
        3. 增强阶段：将检索到的内容作为上下文提供给LLM
        4. 生成阶段：LLM基于查询和上下文生成回答
        
        RAG的优势在于：
        - 减少幻觉（hallucination）问题
        - 提供最新信息，不受预训练数据时间限制
        - 提高回答的可靠性和可溯源性
        - 允许使用私有或专有知识
        
        RAG已成为构建可靠AI应用的重要技术路线。
        """,
        "metadata": {
            "source": "技术博客",
            "author": "AI架构师",
            "category": "人工智能",
            "tags": ["RAG", "检索增强生成", "LLM应用"]
        }
    },
    {
        "title": "向量嵌入技术",
        "content": """
        向量嵌入是将文本、图像、音频等非结构化数据转换为高维数字向量的表示方法。这些向量捕获了原始数据的语义信息，
        使得语义相似的内容在向量空间中也相互接近。
        
        常见的文本嵌入技术包括：
        1. Word2Vec：基于词的上下文学习词向量
        2. GloVe：结合全局矩阵分解和局部上下文窗口
        3. FastText：考虑子词信息的词嵌入
        4. BERT嵌入：利用预训练语言模型生成上下文感知的嵌入
        5. Sentence-BERT：专为句子级别相似度优化的BERT变体
        
        向量嵌入的质量直接影响下游任务的性能，如搜索、推荐、分类等。选择合适的嵌入模型对于特定应用场景至关重要。
        """,
        "metadata": {
            "source": "学术论文",
            "author": "机器学习研究员",
            "category": "机器学习",
            "tags": ["向量嵌入", "文本表示", "语义相似度"]
        }
    },
    {
        "title": "中文自然语言处理挑战",
        "content": """
        中文自然语言处理面临一系列独特的挑战，这些挑战源于中文语言的特性和复杂性。
        
        主要挑战包括：
        1. 分词难题：中文没有明确的词语边界，需要专门的分词算法
        2. 字符编码：处理大量汉字和符号
        3. 语义理解：处理同音词、多义词和文化背景依赖的表达
        4. 方言和变体：处理不同地区的语言变体
        5. 语法结构：中文的语法结构与印欧语系差异较大
        
        针对这些挑战，研究人员开发了专门的中文NLP模型和技术，如中文BERT、ERNIE等预训练模型，
        以及专门的中文分词工具和语义理解框架。这些技术的进步大大提高了中文NLP应用的性能和可用性。
        """,
        "metadata": {
            "source": "研究报告",
            "author": "中文NLP专家",
            "category": "自然语言处理",
            "tags": ["中文NLP", "分词", "语义理解"]
        }
    }
]

async def process_documents(documents: List[Dict], collection_name: str, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    处理文档并存储到向量数据库
    
    Args:
        documents: 文档列表
        collection_name: 集合名称
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
    """
    print(f"初始化服务...")
    start_time = time.time()
    vector_db = VectorDBService()
    embedding_service = EmbeddingService()
    init_time = time.time() - start_time
    print(f"服务初始化完成，耗时: {init_time:.2f}秒")
    print(f"使用嵌入模型: {settings.EMBEDDING_MODEL}")
    print(f"向量维度: {embedding_service.vector_size}")
    
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
        vector_size=embedding_service.vector_size,
        description="批量文档处理示例集合"
    )
    collection = await vector_db.create_collection(collection_create)
    print(f"集合创建成功: {collection.name}, 向量维度: {collection.vector_size}")
    
    # 初始化文本分割器
    text_splitter = ChineseTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 处理每个文档
    total_chunks = 0
    total_vectors = 0
    total_embedding_time = 0
    total_upload_time = 0
    
    for doc_index, doc in enumerate(documents):
        print(f"\n处理文档 {doc_index+1}/{len(documents)}: {doc['title']}")
        
        # 合并标题和内容
        full_text = f"{doc['title']}\n\n{doc['content']}"
        
        # 分割文本
        chunks = text_splitter.split_text(full_text)
        total_chunks += len(chunks)
        print(f"文档被分割为 {len(chunks)} 个块")
        
        # 生成嵌入向量
        start_time = time.time()
        embeddings = await embedding_service.get_embeddings(chunks)
        embedding_time = time.time() - start_time
        total_embedding_time += embedding_time
        print(f"嵌入向量生成完成，耗时: {embedding_time:.2f}秒")
        
        # 准备向量数据
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # 合并文档元数据和块特定元数据
            metadata = {
                **doc['metadata'],
                "doc_id": f"doc_{doc_index}",
                "chunk_id": i,
                "chunk_index": i,
                "chunk_count": len(chunks)
            }
            
            vectors.append(
                VectorUpload(
                    id=f"doc_{doc_index}_chunk_{i}",
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "metadata": metadata
                    }
                )
            )
        
        # 批量上传向量
        start_time = time.time()
        batch_upload = VectorBatchUpload(vectors=vectors)
        result = await vector_db.add_vectors(collection_name, batch_upload)
        upload_time = time.time() - start_time
        total_upload_time += upload_time
        total_vectors += result.added_count
        
        print(f"添加了 {result.added_count} 个向量，耗时: {upload_time:.2f}秒")
    
    # 打印总结
    print(f"\n批处理总结:")
    print(f"处理了 {len(documents)} 个文档")
    print(f"生成了 {total_chunks} 个文本块")
    print(f"添加了 {total_vectors} 个向量")
    print(f"总嵌入时间: {total_embedding_time:.2f}秒")
    print(f"总上传时间: {total_upload_time:.2f}秒")
    print(f"平均每个文档嵌入时间: {total_embedding_time/len(documents):.2f}秒")
    print(f"平均每个文本块嵌入时间: {total_embedding_time/total_chunks:.4f}秒")
    
    return {
        "collection_name": collection_name,
        "document_count": len(documents),
        "chunk_count": total_chunks,
        "vector_count": total_vectors,
        "embedding_time": total_embedding_time,
        "upload_time": total_upload_time
    }

async def search_documents(collection_name: str, queries: List[str]):
    """
    在向量数据库中搜索文档
    
    Args:
        collection_name: 集合名称
        queries: 查询列表
    """
    print(f"\n执行搜索测试...")
    
    # 初始化服务
    vector_db = VectorDBService()
    embedding_service = EmbeddingService()
    
    for i, query in enumerate(queries):
        print(f"\n查询 {i+1}: {query}")
        
        # 生成查询向量
        start_time = time.time()
        query_vector = await embedding_service.get_embeddings(query)
        query_time = time.time() - start_time
        print(f"查询向量生成完成，耗时: {query_time:.4f}秒")
        
        # 执行搜索
        start_time = time.time()
        search_results = await vector_db.search_by_vector(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        search_time = time.time() - start_time
        print(f"搜索完成，耗时: {search_time:.4f}秒")
        
        # 打印结果
        print(f"搜索结果:")
        for j, result in enumerate(search_results):
            print(f"{j+1}. 相似度: {result.score:.4f}")
            print(f"   文档: {result.payload['metadata']['doc_id']} - {result.payload['metadata']['title'] if 'title' in result.payload['metadata'] else '无标题'}")
            print(f"   来源: {result.payload['metadata']['source']}")
            print(f"   类别: {result.payload['metadata']['category']}")
            print(f"   文本片段: {result.payload['text'][:100]}...")
            print()

async def filter_search_example(collection_name: str):
    """
    展示带过滤条件的搜索
    
    Args:
        collection_name: 集合名称
    """
    print(f"\n带过滤条件的搜索示例...")
    
    # 初始化服务
    vector_db = VectorDBService()
    embedding_service = EmbeddingService()
    
    # 查询
    query = "向量数据库的特点是什么？"
    print(f"查询: {query}")
    
    # 生成查询向量
    query_vector = await embedding_service.get_embeddings(query)
    
    # 1. 基本搜索（无过滤）
    print("\n1. 基本搜索（无过滤）:")
    results = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    
    for i, result in enumerate(results):
        print(f"{i+1}. 相似度: {result.score:.4f}")
        print(f"   类别: {result.payload['metadata']['category']}")
        print(f"   文本片段: {result.payload['text'][:100]}...")
        print()
    
    # 2. 按类别过滤
    print("\n2. 按类别过滤 (category='数据库'):")
    filtered_results = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,
        filter={"metadata.category": "数据库"}
    )
    
    for i, result in enumerate(filtered_results):
        print(f"{i+1}. 相似度: {result.score:.4f}")
        print(f"   类别: {result.payload['metadata']['category']}")
        print(f"   文本片段: {result.payload['text'][:100]}...")
        print()
    
    # 3. 按标签过滤
    print("\n3. 按标签过滤 (tags包含'向量数据库'):")
    tag_filtered_results = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,
        filter={"metadata.tags": "向量数据库"}
    )
    
    for i, result in enumerate(tag_filtered_results):
        print(f"{i+1}. 相似度: {result.score:.4f}")
        print(f"   标签: {result.payload['metadata']['tags']}")
        print(f"   文本片段: {result.payload['text'][:100]}...")
        print()

async def export_collection(collection_name: str, output_path: str):
    """
    导出集合数据到JSON文件
    
    Args:
        collection_name: 集合名称
        output_path: 输出文件路径
    """
    print(f"\n导出集合数据...")
    
    # 初始化服务
    vector_db = VectorDBService()
    
    # 获取集合信息
    collection = await vector_db.get_collection(collection_name)
    print(f"集合: {collection.name}, 向量数: {collection.vector_count}")
    
    # 这里需要实现一个搜索所有向量的功能
    # 由于Qdrant不直接支持获取所有向量，我们使用一个模拟的向量进行搜索，并设置大的limit
    # 注意：这只是一个示例，实际应用中应该实现分页获取
    dummy_vector = [0.1] * collection.vector_size
    all_vectors = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=dummy_vector,
        limit=1000,  # 假设集合中的向量数不超过1000
        with_vector=True
    )
    
    # 准备导出数据
    export_data = {
        "collection_info": {
            "name": collection.name,
            "vector_size": collection.vector_size,
            "vector_count": collection.vector_count,
            "description": collection.description
        },
        "vectors": []
    }
    
    for vector in all_vectors:
        export_data["vectors"].append({
            "id": vector.id,
            "score": vector.score,
            "payload": vector.payload,
            "vector": vector.vector[:10] + ["..."]  # 只保存前10个元素，避免文件过大
        })
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"已导出 {len(export_data['vectors'])} 个向量到文件: {output_path}")

async def main():
    """
    主函数
    """
    print("BERT批量文档处理示例")
    print(f"当前配置的嵌入提供者: {settings.EMBEDDING_PROVIDER}")
    print(f"当前配置的嵌入模型: {settings.EMBEDDING_MODEL}")
    
    # 设置集合名称
    collection_name = "bert_batch_demo"
    
    # 1. 处理文档
    result = await process_documents(
        documents=SAMPLE_DOCUMENTS,
        collection_name=collection_name,
        chunk_size=200,
        chunk_overlap=50
    )
    
    # 2. 执行搜索测试
    test_queries = [
        "向量数据库有哪些特点？",
        "BERT模型的创新点是什么？",
        "RAG技术如何提高大语言模型的回答质量？",
        "中文自然语言处理有哪些独特挑战？",
        "向量嵌入技术有哪些常见方法？"
    ]
    await search_documents(collection_name, test_queries)
    
    # 3. 展示带过滤条件的搜索
    await filter_search_example(collection_name)
    
    # 4. 导出集合数据（可选）
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    await export_collection(collection_name, str(output_dir / f"{collection_name}_export.json"))
    
    # 5. 清理
    print("\n清理测试数据...")
    vector_db = VectorDBService()
    await vector_db.delete_collection(collection_name)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(main())