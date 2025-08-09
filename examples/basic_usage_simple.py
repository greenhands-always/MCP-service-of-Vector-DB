"""
向量数据库服务基本使用示例（简化版）
"""
import asyncio
import os
import sys
import uuid
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db.vector_db_service import VectorDBService
from src.models.collection import CollectionCreate, CollectionInfo
from src.models.vector import VectorUpload, VectorBatchUpload
from src.models.search import SearchResult

async def main():
    """
    主函数
    """
    print("向量数据库服务基本使用示例（简化版）")
    
    # 初始化服务
    vector_db = VectorDBService()
    
    # 创建测试集合
    collection_name = "test_collection"
    vector_size = 4  # 简化的向量维度
    
    # 检查集合是否已存在，如果存在则删除
    try:
        collection = await vector_db.get_collection(collection_name)
        if collection:
            print(f"删除已存在的集合: {collection_name}")
            await vector_db.delete_collection(collection_name)
    except Exception as e:
        print(f"集合不存在或删除失败: {e}")
    
    # 创建新集合
    print(f"创建新集合: {collection_name}")
    collection_create = CollectionCreate(
        name=collection_name,
        vector_size=vector_size,
        description="测试集合"
    )
    collection = await vector_db.create_collection(collection_create)
    print(f"集合创建成功: {collection.name}, 向量维度: {collection.vector_size}")
    
    # 准备测试向量
    print("准备测试向量...")
    vectors = [
        VectorUpload(
            id=str(uuid.uuid4()),
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={
                "text": "向量数据库是一种专门设计用于存储、索引和查询向量嵌入的数据库系统。",
                "source": "示例文档",
                "index": 1
            }
        ),
        VectorUpload(
            id=str(uuid.uuid4()),
            vector=[0.2, 0.3, 0.4, 0.5],
            payload={
                "text": "向量嵌入是将文本、图像、音频等非结构化数据转换为高维数字向量的表示方法。",
                "source": "示例文档",
                "index": 2
            }
        ),
        VectorUpload(
            id=str(uuid.uuid4()),
            vector=[0.3, 0.4, 0.5, 0.6],
            payload={
                "text": "向量数据库的核心功能是高效地执行相似性搜索，如最近邻搜索（k-NN）或近似最近邻搜索（ANN）。",
                "source": "示例文档",
                "index": 3
            }
        )
    ]
    
    # 批量添加向量
    print("批量添加向量...")
    batch_upload = VectorBatchUpload(vectors=vectors)
    result = await vector_db.add_vectors(collection_name, batch_upload)
    print(f"添加了 {result.added_count} 个向量")
    
    # 查询示例
    print("\n执行查询示例:")
    
    # 1. 通过向量查询
    query_vector = [0.15, 0.25, 0.35, 0.45]
    print(f"查询向量: {query_vector}")
    
    search_results = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    
    print("查询结果:")
    for i, result in enumerate(search_results):
        print(f"{i+1}. 相似度: {result.score:.4f}")
        print(f"   文本: {result.payload['text']}")
        print()
    
    # 2. 带过滤条件的查询
    print("\n带过滤条件的查询:")
    filtered_results = await vector_db.search_by_vector(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,
        filter={"source": "示例文档"}
    )
    
    print("过滤后的查询结果:")
    for i, result in enumerate(filtered_results):
        print(f"{i+1}. 相似度: {result.score:.4f}")
        print(f"   文本: {result.payload['text']}")
        print()
    
    # 清理
    print("\n清理测试数据...")
    await vector_db.delete_collection(collection_name)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(main())