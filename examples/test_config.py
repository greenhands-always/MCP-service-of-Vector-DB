"""
测试配置加载
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import settings

def main():
    """主函数"""
    print("测试配置加载")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {Path(__file__).parent.parent}")
    
    # 检查.env文件是否存在
    env_path = Path(__file__).parent.parent / ".env"
    env_bert_path = Path(__file__).parent.parent / ".env.bert"
    
    print(f".env文件是否存在: {env_path.exists()}")
    print(f".env.bert文件是否存在: {env_bert_path.exists()}")
    
    # 打印配置
    print("\n当前配置:")
    print(f"EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
    print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print(f"EMBEDDING_DEVICE: {settings.EMBEDDING_DEVICE}")
    
    # 尝试手动加载.env.bert
    if env_bert_path.exists():
        print("\n尝试手动加载.env.bert文件...")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_bert_path)
        
        # 重新加载配置
        from importlib import reload
        from src.core import config
        reload(config)
        
        print("\n手动加载后的配置:")
        print(f"EMBEDDING_PROVIDER: {config.settings.EMBEDDING_PROVIDER}")
        print(f"EMBEDDING_MODEL: {config.settings.EMBEDDING_MODEL}")
        print(f"EMBEDDING_DEVICE: {config.settings.EMBEDDING_DEVICE}")

if __name__ == "__main__":
    main()