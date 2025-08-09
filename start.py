"""
快速启动脚本，用于启动向量数据库MCP服务
"""
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_docker():
    """检查Docker是否已安装"""
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_qdrant():
    """检查Qdrant是否已运行"""
    import requests
    try:
        response = requests.get("http://localhost:6333/")
        return response.status_code == 200
    except:
        return False

def main():
    """主函数"""
    print("向量数据库MCP服务启动脚本")
    print("=" * 40)
    
    # 检查Docker
    if check_docker():
        print("✓ Docker已安装")
        
        # 使用Docker Compose启动服务
        print("\n正在启动服务...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        # 等待服务启动
        print("\n等待服务启动...")
        for _ in range(10):
            if check_qdrant():
                break
            time.sleep(1)
        
        print("\n服务已启动！")
        print("- MCP服务地址: ws://localhost:8000/mcp")
        print("- REST API地址: http://localhost:8000/api/v1")
        print("- API文档地址: http://localhost:8000/docs")
        
        # 打开API文档
        webbrowser.open("http://localhost:8000/docs")
        
        print("\n按Ctrl+C停止服务")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止服务...")
            subprocess.run(["docker-compose", "down"], check=True)
            print("服务已停止")
    else:
        print("✗ Docker未安装")
        print("请安装Docker和Docker Compose后再运行此脚本")
        print("Docker安装指南: https://docs.docker.com/get-docker/")
        
        # 尝试直接运行Python服务
        print("\n尝试直接启动Python服务...")
        
        # 检查是否已安装依赖
        try:
            import fastapi
            import uvicorn
            import qdrant_client
            import sentence_transformers
            
            print("✓ 依赖已安装")
            
            # 检查Qdrant是否已运行
            if not check_qdrant():
                print("✗ Qdrant未运行")
                print("请先启动Qdrant服务")
                print("Qdrant安装指南: https://qdrant.tech/documentation/install/")
                return
            
            # 启动服务
            print("\n正在启动MCP服务...")
            os.environ["QDRANT_HOST"] = "localhost"
            os.environ["QDRANT_PORT"] = "6333"
            
            # 导入并启动服务
            sys.path.insert(0, str(Path(__file__).parent))
            from src.main import start
            start()
            
        except ImportError as e:
            print(f"✗ 缺少依赖: {e}")
            print("请安装依赖后再运行此脚本:")
            print("pip install -e .")

if __name__ == "__main__":
    main()