"""
嵌入服务模块，用于文本向量化
"""
from typing import List, Union, Dict, Any
import numpy as np
import httpx
import asyncio
from loguru import logger

from src.core.config import settings

class EmbeddingService:
    """
    嵌入服务类，用于将文本转换为向量
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化嵌入服务
        
        Args:
            model_name: 模型名称，默认使用配置中的EMBEDDING_MODEL
            device: 设备，默认使用配置中的EMBEDDING_DEVICE
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        
        # 根据EMBEDDING_PROVIDER决定使用哪种嵌入服务
        if settings.EMBEDDING_PROVIDER.lower() == "openai":
            # 使用OpenAI兼容的嵌入服务
            self.openai_embedding = OpenAIEmbedding()
            self.vector_size = self.openai_embedding.vector_size
            self.use_openai = True
            self.use_transformer = False
            logger.info(f"使用OpenAI兼容嵌入服务: {self.model_name}, 向量维度: {self.vector_size}")
        elif settings.EMBEDDING_PROVIDER.lower() == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                # 如果model_name不是完整的huggingface模型路径，则使用默认的BERT模型
                if "/" not in self.model_name:
                    self.model_name = "paraphrase-MiniLM-L6-v2"  # 一个轻量级但效果不错的模型
                
                logger.info(f"使用sentence-transformers嵌入服务: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.vector_size = self.model.get_sentence_embedding_dimension()
                logger.info(f"sentence-transformers嵌入服务初始化完成，向量维度: {self.vector_size}")
                self.use_transformer = True
                self.use_openai = False
            except ImportError:
                logger.warning("未安装sentence-transformers，回退到模拟嵌入服务")
                self.vector_size = 768  # 使用固定的向量维度
                self.use_transformer = False
                self.use_openai = False
        else:
            logger.info(f"使用模拟嵌入服务: {self.model_name}")
            self.vector_size = 768  # 使用固定的向量维度
            self.use_transformer = False
            self.use_openai = False
            
        logger.info(f"嵌入服务初始化完成，向量维度: {self.vector_size}")
    
    async def get_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            嵌入向量或向量列表
        """
        if self.use_openai:
            # 使用OpenAI兼容的嵌入服务
            return await self.openai_embedding.get_embeddings(text)
        elif self.use_transformer:
            if isinstance(text, str):
                # 单个文本，使用模型生成向量
                embedding = self.model.encode(text).tolist()
                return embedding
            else:
                # 文本列表，使用模型为每个文本生成向量
                embeddings = self.model.encode(text).tolist()
                return embeddings
        else:
            # 使用模拟嵌入服务
            if isinstance(text, str):
                # 单个文本，生成固定的向量
                embedding = [0.1] * self.vector_size
                return embedding
            else:
                # 文本列表，为每个文本生成固定的向量
                embeddings = []
                for i, t in enumerate(text):
                    # 生成稍微不同的向量
                    embedding = [0.1 * (i + 1) + 0.01 * j for j in range(self.vector_size)]
                    embeddings.append(embedding)
                return embeddings
    
    async def get_embeddings_with_metadata(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        获取文本的嵌入向量，并附加元数据
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表，与texts一一对应
            
        Returns:
            包含向量和元数据的字典
        """
        embeddings = await self.get_embeddings(texts)
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        return {
            "embeddings": embeddings,
            "metadatas": metadatas,
            "texts": texts,
            "model": self.model_name,
            "dimension": self.vector_size
        }


class OpenAIEmbedding:
    """
    OpenAI兼容协议的嵌入服务类
    支持通过API接入的方式进行向量化
    """
    
    def __init__(
        self, 
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        初始化OpenAI兼容的嵌入服务
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key or getattr(settings, 'EMBEDDING_API_KEY', None)
        self.base_url = base_url or getattr(settings, 'EMBEDDING_API_BASE', 'https://api.openai.com/v1')
        self.model_name = model_name or getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 设置向量维度 - 根据不同模型设置默认维度
        model_dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-v4': 1024,  # 阿里云DashScope的text-embedding-v4模型
        }
        self.vector_size = model_dimensions.get(self.model_name, 1536)  # 默认1536维
        
        # 确保base_url以/结尾
        if not self.base_url.endswith('/'):
            self.base_url += '/'
            
        self.embeddings_url = f"{self.base_url}embeddings"
        
        if not self.api_key:
            raise ValueError("API密钥不能为空，请设置OPENAI_API_KEY环境变量或传入api_key参数")
            
        logger.info(f"初始化OpenAI兼容嵌入服务: {self.model_name}, URL: {self.embeddings_url}, 向量维度: {self.vector_size}")
    
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送HTTP请求到OpenAI兼容的API
        
        Args:
            payload: 请求负载
            
        Returns:
            API响应数据
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.embeddings_url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        error_msg = f"API请求失败，状态码: {response.status_code}, 响应: {response.text}"
                        logger.error(error_msg)
                        if attempt == self.max_retries - 1:
                            raise Exception(error_msg)
                        
            except httpx.TimeoutException:
                error_msg = f"请求超时 (尝试 {attempt + 1}/{self.max_retries})"
                logger.warning(error_msg)
                if attempt == self.max_retries - 1:
                    raise Exception(f"请求超时，已重试{self.max_retries}次")
                    
            except Exception as e:
                error_msg = f"请求异常: {str(e)} (尝试 {attempt + 1}/{self.max_retries})"
                logger.error(error_msg)
                if attempt == self.max_retries - 1:
                    raise
                    
            # 等待后重试
            await asyncio.sleep(2 ** attempt)
    
    async def get_embeddings(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            嵌入向量或向量列表
        """
        # 统一处理为列表格式
        texts = [text] if isinstance(text, str) else text
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        try:
            response_data = await self._make_request(payload)
            
            # 提取嵌入向量
            embeddings = []
            for item in response_data.get("data", []):
                embeddings.append(item.get("embedding", []))
            
            # 如果输入是单个文本，返回单个向量
            if isinstance(text, str):
                return embeddings[0] if embeddings else []
            else:
                return embeddings
                
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {str(e)}")
            raise
    
    async def get_embeddings_with_metadata(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        获取文本的嵌入向量，并附加元数据
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表，与texts一一对应
            
        Returns:
            包含向量和元数据的字典
        """
        embeddings = await self.get_embeddings(texts)
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # 获取向量维度
        vector_size = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
        
        return {
            "embeddings": embeddings,
            "metadatas": metadatas,
            "texts": texts,
            "model": self.model_name,
            "dimension": vector_size
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model_name,
            "provider": "openai_compatible",
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }


def create_embedding_service(provider: str = None) -> Union[EmbeddingService, OpenAIEmbedding]:
    """
    工厂函数：根据配置创建相应的嵌入服务实例
    
    Args:
        provider: 嵌入服务提供商，可选值: 'sentence-transformers', 'openai', 'mock'
        
    Returns:
        嵌入服务实例
    """
    provider = provider or getattr(settings, 'EMBEDDING_PROVIDER', 'sentence-transformers')
    
    if provider.lower() == 'openai':
        return OpenAIEmbedding()
    else:
        return EmbeddingService()
