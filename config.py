"""全局配置管理，从环境变量加载所有参数。"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """RAG 系统全局配置。"""

    # OpenAI 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # 模型配置
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    # 向量数据库配置
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_knowledge_base")

    # 文本分块配置
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # 检索配置
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    RETRIEVAL_STRATEGY: str = os.getenv("RETRIEVAL_STRATEGY", "mmr")

    @classmethod
    def validate(cls) -> None:
        """校验必要配置是否存在。"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "未设置 OPENAI_API_KEY，请复制 .env.example 为 .env 并填入你的 API Key。"
            )
