"""
OpenAI兼容协议嵌入服务使用示例
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.embedding import OpenAIEmbedding, create_embedding_service
from src.core.config import settings
from loguru import logger

async def test_openai_embedding():
    """测试OpenAI兼容协议的嵌入服务"""
    
    # 方式1: 直接创建OpenAI嵌入服务实例
    print("=== 方式1: 直接创建OpenAI嵌入服务实例 ===")
    
    # 配置API参数（请替换为实际的API信息）
    # api_key = "your-api-key-here"  # 替换为实际的API密钥
    # base_url = "https://api.openai.com/v1"  # 或其他兼容OpenAI的API地址
    # model_name = "text-embedding-ada-002"  # 或其他支持的模型
    #
    try:
        # 创建OpenAI嵌入服务实例
        embedding_service = OpenAIEmbedding(
            # api_key=api_key,
            # base_url=base_url,
            # model_name=model_name,
            timeout=30,
            max_retries=3
        )
        
        # 测试单个文本的嵌入
        print("测试单个文本嵌入...")
        single_text = "这是一个测试文本"
        single_embedding = await embedding_service.get_embeddings(single_text)
        print(f"单个文本嵌入维度: {len(single_embedding)}")
        print(f"前5个向量值: {single_embedding[:5]}")
        
        # 测试多个文本的嵌入
        print("\n测试多个文本嵌入...")
        texts = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的核心技术",
            "深度学习是机器学习的一个子领域"
        ]
        
        embeddings = await embedding_service.get_embeddings(texts)
        print(f"批量文本嵌入数量: {len(embeddings)}")
        for i, embedding in enumerate(embeddings):
            print(f"文本{i+1}嵌入维度: {len(embedding)}")
        
        # 测试带元数据的嵌入
        print("\n测试带元数据的嵌入...")
        metadatas = [
            {"category": "AI", "source": "textbook"},
            {"category": "ML", "source": "paper"},
            {"category": "DL", "source": "article"}
        ]
        
        result = await embedding_service.get_embeddings_with_metadata(texts, metadatas)
        print(f"结果包含的键: {list(result.keys())}")
        print(f"模型名称: {result['model']}")
        print(f"向量维度: {result['dimension']}")
        
        # 获取模型信息
        print("\n获取模型信息...")
        model_info = await embedding_service.get_model_info()
        print(f"模型信息: {model_info}")
        
    except Exception as e:
        print(f"OpenAI嵌入服务测试失败: {str(e)}")
        print("请检查API密钥和网络连接")

async def test_factory_function():
    """测试工厂函数创建嵌入服务"""
    
    print("\n=== 方式2: 使用工厂函数创建嵌入服务 ===")
    
    # 设置环境变量（在实际使用中，这些应该在.env文件中设置）
    os.environ['EMBEDDING_PROVIDER'] = 'openai'
    os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'
    os.environ['OPENAI_EMBEDDING_MODEL'] = 'text-embedding-ada-002'
    
    try:
        # 使用工厂函数创建嵌入服务
        embedding_service = create_embedding_service('openai')
        
        # 测试嵌入功能
        test_text = "使用工厂函数创建的嵌入服务测试"
        embedding = await embedding_service.get_embeddings(test_text)
        print(f"工厂函数创建的服务嵌入维度: {len(embedding)}")
        
    except Exception as e:
        print(f"工厂函数测试失败: {str(e)}")

async def test_different_providers():
    """测试不同的嵌入服务提供商"""
    
    print("\n=== 测试不同的嵌入服务提供商 ===")
    
    providers = ['sentence-transformers', 'openai']
    test_text = "这是一个用于测试不同提供商的文本"
    
    for provider in providers:
        print(f"\n测试提供商: {provider}")
        try:
            embedding_service = create_embedding_service(provider)
            embedding = await embedding_service.get_embeddings(test_text)
            print(f"{provider} 嵌入维度: {len(embedding)}")
            
        except Exception as e:
            print(f"{provider} 测试失败: {str(e)}")


async def main():
    """主函数"""
    
    print("OpenAI兼容协议嵌入服务测试")
    print("=" * 50)

    
    # 测试OpenAI嵌入服务
    await test_openai_embedding()
    
    # 测试工厂函数
    await test_factory_function()
    
    # 测试不同提供商
    await test_different_providers()
    
    print("\n测试完成!")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())