"""文档加载器：支持 PDF、Markdown、TXT 等多种格式。"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document


class DocumentLoader:
    """统一文档加载接口，自动根据文件扩展名选择加载器。"""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文档文件。

        Args:
            file_path: 文档文件路径

        Returns:
            Document 对象列表

        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {ext}，支持格式: {self.SUPPORTED_EXTENSIONS}"
            )

        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding=self.encoding)

        documents = loader.load()
        # 为每个文档添加来源元数据
        for doc in documents:
            doc.metadata["source"] = str(path)
            doc.metadata["filename"] = path.name

        return documents

    def load_directory(self, dir_path: str, glob: str = "**/*.*") -> List[Document]:
        """批量加载目录下所有支持的文档。

        Args:
            dir_path: 目录路径
            glob: 文件匹配模式

        Returns:
            Document 对象列表
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        all_docs: List[Document] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            try:
                loader = DirectoryLoader(
                    str(path),
                    glob=f"**/*{ext}",
                    loader_cls=TextLoader if ext != ".pdf" else PyPDFLoader,
                    loader_kwargs={"encoding": self.encoding} if ext != ".pdf" else {},
                    show_progress=True,
                )
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"加载 {ext} 文件时出错: {e}")

        return all_docs

    def load_text(self, text: str, source: str = "direct_input") -> List[Document]:
        """直接加载文本内容。

        Args:
            text: 文本内容
            source: 来源标识

        Returns:
            包含单个 Document 的列表
        """
        doc = Document(page_content=text, metadata={"source": source})
        return [doc]
