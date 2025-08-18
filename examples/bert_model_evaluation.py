"""
BERT模型评估脚本 - 比较不同BERT模型的性能和效果
"""
import asyncio
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.embedding import EmbeddingService
from src.core.config import settings

# 测试数据
TEST_TEXTS = [
    "向量数据库是一种专门设计用于存储、索引和查询向量嵌入的数据库系统。",
    "向量嵌入是将文本、图像、音频等非结构化数据转换为高维数字向量的表示方法。",
    "BERT是一种预训练的语言模型，能够生成上下文感知的文本表示。",
    "RAG（检索增强生成）是一种结合了检索系统和生成式AI的技术架构。",
    "在RAG架构中，向量数据库扮演着存储知识库、语义检索和上下文增强的关键角色。"
]

TEST_QUERIES = [
    "什么是向量数据库？",
    "向量嵌入的作用是什么？",
    "BERT模型有什么特点？",
    "RAG技术的工作原理是什么？",
    "向量数据库在RAG中的作用是什么？"
]

# 要评估的模型列表
MODELS_TO_EVALUATE = [
    "paraphrase-MiniLM-L6-v2",  # 轻量级英文模型
    "all-MiniLM-L6-v2",         # 另一个轻量级英文模型
    "BAAI/bge-small-zh",        # 轻量级中文模型
    "shibing624/text2vec-base-chinese"  # 中文语义相似度模型
]

def cosine_similarity(v1, v2):
    """计算两个向量之间的余弦相似度"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    return dot_product / (norm_v1 * norm_v2)

async def evaluate_model(model_name: str):
    """评估单个模型的性能"""
    print(f"\n{'='*80}")
    print(f"评估模型: {model_name}")
    print(f"{'='*80}")
    
    # 设置环境变量
    os.environ["EMBEDDING_PROVIDER"] = "sentence-transformers"
    os.environ["EMBEDDING_MODEL"] = model_name
    os.environ["EMBEDDING_DEVICE"] = "cpu"
    
    try:
        # 初始化嵌入服务
        start_time = time.time()
        embedding_service = EmbeddingService()
        init_time = time.time() - start_time
        
        if not hasattr(embedding_service, 'use_transformer') or not embedding_service.use_transformer:
            print(f"模型 {model_name} 加载失败，可能是模型不存在或sentence-transformers未安装")
            return {
                "model": model_name,
                "status": "failed",
                "error": "模型加载失败"
            }
        
        print(f"模型加载时间: {init_time:.2f}秒")
        print(f"向量维度: {embedding_service.vector_size}")
        
        # 测试单个文本嵌入性能
        start_time = time.time()
        single_embedding = await embedding_service.get_embeddings(TEST_TEXTS[0])
        single_time = time.time() - start_time
        print(f"单个文本嵌入时间: {single_time:.4f}秒")
        
        # 测试批量文本嵌入性能
        start_time = time.time()
        batch_embeddings = await embedding_service.get_embeddings(TEST_TEXTS)
        batch_time = time.time() - start_time
        print(f"批量文本嵌入时间 ({len(TEST_TEXTS)}个文本): {batch_time:.4f}秒")
        print(f"平均每个文本: {batch_time/len(TEST_TEXTS):.4f}秒")
        
        # 测试查询嵌入性能
        start_time = time.time()
        query_embeddings = await embedding_service.get_embeddings(TEST_QUERIES)
        query_time = time.time() - start_time
        print(f"查询嵌入时间 ({len(TEST_QUERIES)}个查询): {query_time:.4f}秒")
        
        # 计算相似度矩阵
        similarity_matrix = []
        for q_emb in query_embeddings:
            row = []
            for t_emb in batch_embeddings:
                sim = cosine_similarity(q_emb, t_emb)
                row.append(sim)
            similarity_matrix.append(row)
        
        # 计算平均相似度
        avg_similarity = sum(sum(row) for row in similarity_matrix) / (len(similarity_matrix) * len(similarity_matrix[0]))
        print(f"平均相似度: {avg_similarity:.4f}")
        
        # 计算相关查询的平均相似度
        relevant_similarities = []
        for i in range(len(TEST_QUERIES)):
            relevant_similarities.append(similarity_matrix[i][i])
        avg_relevant_similarity = sum(relevant_similarities) / len(relevant_similarities)
        print(f"相关查询平均相似度: {avg_relevant_similarity:.4f}")
        
        # 计算非相关查询的平均相似度
        irrelevant_similarities = []
        for i in range(len(TEST_QUERIES)):
            for j in range(len(TEST_TEXTS)):
                if i != j:
                    irrelevant_similarities.append(similarity_matrix[i][j])
        avg_irrelevant_similarity = sum(irrelevant_similarities) / len(irrelevant_similarities)
        print(f"非相关查询平均相似度: {avg_irrelevant_similarity:.4f}")
        
        # 计算对比度 (相关与非相关的差异)
        contrast = avg_relevant_similarity - avg_irrelevant_similarity
        print(f"相关性对比度: {contrast:.4f}")
        
        # 打印相似度矩阵
        print("\n相似度矩阵:")
        print("查询 \\ 文本", end="\t")
        for i in range(len(TEST_TEXTS)):
            print(f"文本{i+1}", end="\t")
        print()
        
        for i in range(len(TEST_QUERIES)):
            print(f"查询{i+1}", end="\t")
            for j in range(len(TEST_TEXTS)):
                print(f"{similarity_matrix[i][j]:.4f}", end="\t")
            print()
        
        return {
            "model": model_name,
            "status": "success",
            "vector_size": embedding_service.vector_size,
            "init_time": init_time,
            "single_time": single_time,
            "batch_time": batch_time,
            "query_time": query_time,
            "avg_similarity": avg_similarity,
            "avg_relevant_similarity": avg_relevant_similarity,
            "avg_irrelevant_similarity": avg_irrelevant_similarity,
            "contrast": contrast
        }
    
    except Exception as e:
        print(f"评估模型 {model_name} 时出错: {str(e)}")
        return {
            "model": model_name,
            "status": "error",
            "error": str(e)
        }

async def main():
    """主函数"""
    print("BERT模型评估")
    print("这个脚本将评估不同BERT模型的性能和效果")
    
    # 保存原始环境变量
    original_provider = os.environ.get("EMBEDDING_PROVIDER")
    original_model = os.environ.get("EMBEDDING_MODEL")
    original_device = os.environ.get("EMBEDDING_DEVICE")
    
    results = []
    
    # 评估每个模型
    for model in MODELS_TO_EVALUATE:
        try:
            result = await evaluate_model(model)
            results.append(result)
        except Exception as e:
            print(f"评估模型 {model} 时发生异常: {str(e)}")
            results.append({
                "model": model,
                "status": "exception",
                "error": str(e)
            })
    
    # 恢复原始环境变量
    if original_provider:
        os.environ["EMBEDDING_PROVIDER"] = original_provider
    if original_model:
        os.environ["EMBEDDING_MODEL"] = original_model
    if original_device:
        os.environ["EMBEDDING_DEVICE"] = original_device
    
    # 打印比较结果
    print("\n\n模型比较结果:")
    print(f"{'模型名称':<40} {'向量维度':<10} {'初始化时间(s)':<15} {'单文本时间(s)':<15} {'批处理时间(s)':<15} {'相关性对比度':<15}")
    print("-" * 110)
    
    for result in results:
        if result["status"] == "success":
            print(f"{result['model']:<40} {result['vector_size']:<10} {result['init_time']:<15.4f} {result['single_time']:<15.4f} {result['batch_time']:<15.4f} {result['contrast']:<15.4f}")
        else:
            print(f"{result['model']:<40} {'N/A':<10} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} - {result['error']}")
    
    # 找出最佳模型
    success_results = [r for r in results if r["status"] == "success"]
    if success_results:
        # 按相关性对比度排序
        best_contrast = max(success_results, key=lambda x: x["contrast"])
        # 按批处理速度排序
        fastest_batch = min(success_results, key=lambda x: x["batch_time"])
        # 按初始化时间排序
        fastest_init = min(success_results, key=lambda x: x["init_time"])
        
        print("\n最佳模型推荐:")
        print(f"最佳相关性: {best_contrast['model']} (对比度: {best_contrast['contrast']:.4f})")
        print(f"最快批处理: {fastest_batch['model']} (时间: {fastest_batch['batch_time']:.4f}秒)")
        print(f"最快初始化: {fastest_init['model']} (时间: {fastest_init['init_time']:.4f}秒)")
    
    print("\n评估完成")

if __name__ == "__main__":
    asyncio.run(main())