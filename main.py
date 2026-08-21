"""RAG 知识库问答系统 - 主程序入口。

支持两种运行模式：
1. CLI 交互模式：python main.py chat
2. 文档索引模式：python main.py index <文件或目录路径>
3. Web UI 模式：python main.py web
"""

import argparse
import sys
from pathlib import Path

from config import Config
from src import DocumentLoader, TextChunker, VectorStoreManager, Retriever, RAGChain


class RAGApplication:
    """RAG 应用主类，整合所有模块。"""

    def __init__(self):
        Config.validate()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        self.vector_store = VectorStoreManager()
        self.retriever = Retriever(self.vector_store)
        self.rag_chain = RAGChain(self.retriever)

    def index_documents(self, path: str) -> int:
        """索引文档到向量库。

        Args:
            path: 文件或目录路径

        Returns:
            索引的文档块数量
        """
        p = Path(path)
        if not p.exists():
            print(f"错误: 路径不存在 - {path}")
            return 0

        print(f"正在加载文档: {path}")
        if p.is_file():
            documents = self.loader.load_file(str(p))
        else:
            documents = self.loader.load_directory(str(p))

        print(f"加载了 {len(documents)} 个文档段")

        print("正在分块...")
        chunks = self.chunker.split_documents(documents)
        print(f"生成了 {len(chunks)} 个文本块")

        print("正在向量化并存储...")
        ids = self.vector_store.add_documents(chunks)
        print(f"成功索引 {len(ids)} 个文档块")

        return len(ids)

    def chat(self):
        """启动交互式问答。"""
        print("=" * 60)
        print("RAG 知识库问答系统")
        print(f"向量库文档数: {self.vector_store.count()}")
        print(f"检索策略: {Config.RETRIEVAL_STRATEGY}")
        print("输入 'quit' 或 'exit' 退出，输入 'stats' 查看状态")
        print("=" * 60)

        while True:
            try:
                question = input("\n请输入问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if question.lower() == "stats":
                print(f"向量库文档数: {self.vector_store.count()}")
                print(f"当前检索策略: {Config.RETRIEVAL_STRATEGY}")
                continue

            print("\n正在检索和生成回答...")
            result = self.rag_chain.query(question)
            print(RAGChain.format_result(result))

    def query_once(self, question: str):
        """单次查询。"""
        result = self.rag_chain.query(question)
        print(RAGChain.format_result(result))

    def stats(self):
        """显示系统状态。"""
        print(f"向量库路径: {Config.CHROMA_PERSIST_DIR}")
        print(f"集合名称: {Config.COLLECTION_NAME}")
        print(f"文档块数量: {self.vector_store.count()}")
        print(f"嵌入模型: {Config.EMBEDDING_MODEL}")
        print(f"对话模型: {Config.CHAT_MODEL}")
        print(f"分块大小: {Config.CHUNK_SIZE} (重叠 {Config.CHUNK_OVERLAP})")
        print(f"Top-K: {Config.TOP_K}")
        print(f"检索策略: {Config.RETRIEVAL_STRATEGY}")


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库问答系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # index 命令
    index_parser = subparsers.add_parser("index", help="索引文档")
    index_parser.add_argument("path", help="文档文件或目录路径")

    # chat 命令
    subparsers.add_parser("chat", help="启动交互式问答")

    # query 命令
    query_parser = subparsers.add_parser("query", help="单次查询")
    query_parser.add_argument("question", help="问题文本")

    # stats 命令
    subparsers.add_parser("stats", help="显示系统状态")

    # web 命令
    subparsers.add_parser("web", help="启动 Web UI (Streamlit)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    app = RAGApplication()

    if args.command == "index":
        app.index_documents(args.path)
    elif args.command == "chat":
        app.chat()
    elif args.command == "query":
        app.query_once(args.question)
    elif args.command == "stats":
        app.stats()
    elif args.command == "web":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "web_ui.py"])


if __name__ == "__main__":
    main()
