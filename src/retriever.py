"""检索器：支持相似度、MMR、混合检索等多种策略。"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from .vector_store import VectorStoreManager
from config import Config


class Retriever:
    """统一检索接口，支持多种检索策略切换。"""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self._bm25_retriever: Optional[BM25Retriever] = None

    def retrieve(
        self,
        query: str,
        strategy: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        """执行检索。

        Args:
            query: 查询文本
            strategy: 检索策略 (similarity / mmr / ensemble)
            k: 返回结果数量

        Returns:
            相关文档列表
        """
        strategy = strategy or Config.RETRIEVAL_STRATEGY
        k = k or Config.TOP_K

        if strategy == "similarity":
            return self._similarity_retrieve(query, k)
        elif strategy == "mmr":
            return self._mmr_retrieve(query, k)
        elif strategy == "ensemble":
            return self._ensemble_retrieve(query, k)
        else:
            raise ValueError(f"不支持的检索策略: {strategy}")

    def _similarity_retrieve(self, query: str, k: int) -> List[Document]:
        """纯向量相似度检索。"""
        return self.vector_store.similarity_search(query, k=k)

    def _mmr_retrieve(self, query: str, k: int) -> List[Document]:
        """最大边际相关性检索，兼顾相关性和多样性。"""
        return self.vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=min(k * 5, 50), lambda_mult=0.5
        )

    def _ensemble_retrieve(self, query: str, k: int) -> List[Document]:
        """混合检索：向量相似度 + BM25 关键词检索。"""
        vector_retriever = self.vector_store.get_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

        # 从向量库获取所有文档构建 BM25 索引
        if self._bm25_retriever is None:
            all_docs = self._get_all_documents()
            if all_docs:
                self._bm25_retriever = BM25Retriever.from_documents(all_docs)
                self._bm25_retriever.k = k
            else:
                return vector_retriever.get_relevant_documents(query)

        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, self._bm25_retriever],
            weights=[0.6, 0.4],
        )
        return ensemble.get_relevant_documents(query)

    def _get_all_documents(self) -> List[Document]:
        """从向量库获取所有文档（用于构建 BM25 索引）。"""
        try:
            collection = self.vector_store.vector_store._collection
            results = collection.get(include=["documents", "metadatas"])
            docs = []
            for content, meta in zip(results["documents"], results["metadatas"]):
                docs.append(Document(page_content=content, metadata=meta or {}))
            return docs
        except Exception:
            return []

    def get_base_retriever(self, strategy: str = "mmr") -> BaseRetriever:
        """获取 LangChain 原生 Retriever，用于链组合。"""
        if strategy == "ensemble":
            return self._build_ensemble_retriever()
        return self.vector_store.get_retriever(search_type=strategy)

    def _build_ensemble_retriever(self) -> EnsembleRetriever:
        """构建混合检索器。"""
        vector_retriever = self.vector_store.get_retriever(
            search_type="similarity", search_kwargs={"k": Config.TOP_K}
        )
        all_docs = self._get_all_documents()
        bm25 = BM25Retriever.from_documents(all_docs) if all_docs else None
        bm25.k = Config.TOP_K if bm25 else None
        if bm25:
            return EnsembleRetriever(
                retrievers=[vector_retriever, bm25], weights=[0.6, 0.4]
            )
        return vector_retriever
