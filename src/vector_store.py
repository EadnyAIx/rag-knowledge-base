"""向量存储管理器：基于 ChromaDB 的持久化向量数据库。"""

from typing import List, Optional

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import Config


class VectorStoreManager:
    """ChromaDB 向量存储管理，支持增删查和持久化。"""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """初始化向量存储。

        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
        """
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or Config.COLLECTION_NAME

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model or Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """向向量库添加文档。

        Args:
            documents: 文档块列表

        Returns:
            文档 ID 列表
        """
        if not documents:
            return []

        ids = self.vector_store.add_documents(documents)
        return ids

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """直接向向量库添加文本。

        Args:
            texts: 文本列表
            metadatas: 元数据列表

        Returns:
            文档 ID 列表
        """
        ids = self.vector_store.add_texts(texts, metadatas=metadatas)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> List[Document]:
        """相似度搜索。

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            相关文档列表
        """
        return self.vector_store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[tuple]:
        """带分数的相似度搜索。

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            (Document, score) 元组列表
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> List[Document]:
        """最大边际相关性搜索（MMR），兼顾相关性与多样性。

        Args:
            query: 查询文本
            k: 返回结果数量
            fetch_k: 候选集大小
            lambda_mult: 相关性-多样性权衡参数 (0=最大多样性, 1=最大相关性)

        Returns:
            相关文档列表
        """
        return self.vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """删除文档。

        Args:
            ids: 要删除的文档 ID 列表，为空则清空集合
        """
        if ids:
            self.vector_store.delete(ids=ids)
        else:
            # 清空整个集合
            client = chromadb.PersistentClient(path=self.persist_dir)
            client.delete_collection(self.collection_name)
            # 重新创建
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )

    def count(self) -> int:
        """返回向量库中文档数量。"""
        return self.vector_store._collection.count()

    def get_retriever(self, search_type: str = "mmr", search_kwargs: Optional[dict] = None):
        """获取 LangChain Retriever 对象。

        Args:
            search_type: 搜索类型 (similarity / mmr)
            search_kwargs: 搜索参数

        Returns:
            VectorStoreRetriever 对象
        """
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": Config.TOP_K},
        )
