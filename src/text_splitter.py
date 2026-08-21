"""文本分块器：支持按字符、递归字符、语义等分块策略。"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


class TextChunker:
    """文本分块管理器，支持多种分块策略。"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "recursive",
    ):
        """初始化分块器。

        Args:
            chunk_size: 每块最大字符数
            chunk_overlap: 块之间重叠字符数
            strategy: 分块策略 (recursive / character)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        if strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            )
        elif strategy == "character":
            self.splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n",
            )
        else:
            raise ValueError(f"不支持的分块策略: {strategy}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档列表切分为文本块。

        Args:
            documents: 原始文档列表

        Returns:
            切分后的文档块列表
        """
        chunks = self.splitter.split_documents(documents)

        # 为每个块添加分块元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """直接切分纯文本。

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        return self.splitter.split_text(text)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的 token 数量（粗略估算，中文约 1.7 字符/token）。"""
        return max(1, int(len(text) / 1.7))
