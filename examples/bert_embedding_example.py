"""
BERT向量化示例 - 展示如何使用BERT模型进行文本向量化
"""
import asyncio
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.embedding import EmbeddingService
from src.core.config import settings

async def main():
    """
    主函数
    """
    print("BERT向量化示例")
    print(f"当前配置的嵌入提供者: {settings.EMBEDDING_PROVIDER}")
    print(f"当前配置的嵌入模型: {settings.EMBEDDING_MODEL}")
    print(f"当前配置的设备: {settings.EMBEDDING_DEVICE}")
    
    # 初始化嵌入服务
    print("\n初始化嵌入服务...")
    start_time = time.time()
    embedding_service = EmbeddingService()
    init_time = time.time() - start_time
    print(f"嵌入服务初始化完成，耗时: {init_time:.2f}秒")
    print(f"向量维度: {embedding_service.vector_size}")
    
    # 准备测试文本
    print("\n准备测试文本...")
    texts = [
        "向量数据库是一种专门设计用于存储、索引和查询向量嵌入的数据库系统。",
        "向量嵌入是将文本、图像、音频等非结构化数据转换为高维数字向量的表示方法。",
        "BERT（Bidirectional Encoder Representations from Transformers）是一种预训练的语言模型。",
        "RAG（检索增强生成）是一种结合了检索系统和生成式AI的技术架构。",
        "在RAG架构中，向量数据库扮演着存储知识库、语义检索和上下文增强的关键角色。"
    ]
    
    # 生成单个文本的嵌入
    print("\n生成单个文本的嵌入...")
    start_time = time.time()
    single_embedding = await embedding_service.get_embeddings(texts[0])
    single_time = time.time() - start_time
    print(f"单个文本嵌入生成完成，耗时: {single_time:.2f}秒")
    print(f"向量维度: {len(single_embedding)}")
    print(f"向量前10个元素: {single_embedding[:10]}")
    
    # 生成批量文本的嵌入
    print("\n生成批量文本的嵌入...")
    start_time = time.time()
    batch_embeddings = await embedding_service.get_embeddings(texts)
    batch_time = time.time() - start_time
    print(f"批量文本嵌入生成完成，耗时: {batch_time:.2f}秒")
    print(f"生成了 {len(batch_embeddings)} 个向量")
    print(f"每个向量的维度: {len(batch_embeddings[0])}")
    
    # 计算向量之间的相似度
    print("\n计算向量之间的余弦相似度:")
    
    def cosine_similarity(v1, v2):
        """计算两个向量之间的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(a * a for a in v1) ** 0.5
        norm_v2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (norm_v1 * norm_v2)
    
    # 创建相似度矩阵
    similarity_matrix = []
    for i, emb1 in enumerate(batch_embeddings):
        row = []
        for j, emb2 in enumerate(batch_embeddings):
            sim = cosine_similarity(emb1, emb2)
            row.append(sim)
        similarity_matrix.append(row)
    
    # 打印相似度矩阵
    print("\n相似度矩阵:")
    for i in range(len(texts)):
        for j in range(len(texts)):
            print(f"{similarity_matrix[i][j]:.4f}", end="\t")
        print()
    
    # 找出最相似的文本对
    max_sim = 0
    max_i, max_j = 0, 0
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):  # 避免自身比较和重复比较
            if similarity_matrix[i][j] > max_sim:
                max_sim = similarity_matrix[i][j]
                max_i, max_j = i, j
    
    print(f"\n最相似的文本对 (相似度: {max_sim:.4f}):")
    print(f"文本1: {texts[max_i]}")
    print(f"文本2: {texts[max_j]}")
    
    # 测试查询相似度
    print("\n测试查询相似度:")
    query = "向量数据库在RAG中的作用是什么？"
    print(f"查询: {query}")
    
    # 生成查询向量
    start_time = time.time()
    query_embedding = await embedding_service.get_embeddings(query)
    query_time = time.time() - start_time
    print(f"查询向量生成完成，耗时: {query_time:.2f}秒")
    
    # 计算查询与所有文本的相似度
    similarities = []
    for i, emb in enumerate(batch_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 打印排序后的结果
    print("\n查询结果 (按相似度排序):")
    for i, (idx, sim) in enumerate(similarities):
        print(f"{i+1}. 相似度: {sim:.4f}")
        print(f"   文本: {texts[idx]}")
        print()
    
    # 测试带元数据的嵌入
    print("\n测试带元数据的嵌入:")
    metadatas = [{"source": f"文档{i}", "category": "技术文档"} for i in range(len(texts))]
    
    start_time = time.time()
    result = await embedding_service.get_embeddings_with_metadata(texts, metadatas)
    metadata_time = time.time() - start_time
    print(f"带元数据的嵌入生成完成，耗时: {metadata_time:.2f}秒")
    print(f"结果包含 {len(result['embeddings'])} 个向量")
    print(f"模型名称: {result['model']}")
    print(f"向量维度: {result['dimension']}")
    
    print("\n测试完成")

if __name__ == "__main__":
    asyncio.run(main())