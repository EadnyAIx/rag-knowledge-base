"""RAG 问答链：整合检索与生成，支持引用溯源和流式输出。"""

from typing import List, Optional, Dict, Any, Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from .retriever import Retriever
from config import Config


# RAG 系统提示词
RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请基于以下检索到的上下文信息回答用户的问题。

【回答规则】
1. 严格基于提供的上下文内容回答，不要编造上下文之外的信息
2. 如果上下文中没有相关信息，请明确回答"根据现有知识库，无法找到相关信息"
3. 回答要简洁、准确、有条理
4. 引用来源时使用 [来源:文件名] 格式标注

【检索到的上下文】
{context}

【用户问题】
{question}

请给出你的回答："""


class RAGChain:
    """RAG 问答链，整合检索与生成。"""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

        self.llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            temperature=0.1,
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
        self._chain = self._build_chain()

    def _build_chain(self):
        """构建 RAG 处理链。"""
        base_retriever = self.retriever.get_base_retriever(Config.RETRIEVAL_STRATEGY)

        def format_docs(docs: List[Document]) -> str:
            """格式化检索到的文档。"""
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("filename", doc.metadata.get("source", "未知"))
                formatted.append(f"--- 文档 {i} (来源: {source}) ---\n{doc.page_content}")
            return "\n\n".join(formatted)

        chain = (
            {
                "context": base_retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """执行 RAG 查询。

        Args:
            question: 用户问题
            strategy: 检索策略覆盖

        Returns:
            包含回答、来源文档、检索策略的字典
        """
        # 先执行检索获取来源文档
        retrieved_docs = self.retriever.retrieve(
            question, strategy=strategy, k=Config.TOP_K
        )

        # 执行生成
        answer = self._chain.invoke(question)

        # 整理来源信息
        sources = []
        seen = set()
        for doc in retrieved_docs:
            source = doc.metadata.get("filename", doc.metadata.get("source", "未知"))
            if source not in seen:
                seen.add(source)
                sources.append({
                    "source": source,
                    "content_preview": doc.page_content[:200],
                    "relevance": "高",
                })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(retrieved_docs),
            "strategy": strategy or Config.RETRIEVAL_STRATEGY,
        }

    def stream_query(self, question: str, strategy: Optional[str] = None) -> Iterator[str]:
        """流式输出 RAG 查询结果。

        Args:
            question: 用户问题
            strategy: 检索策略

        Yields:
            逐 token 的回答文本
        """
        for chunk in self._chain.stream(question):
            yield chunk

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量查询。

        Args:
            questions: 问题列表

        Returns:
            结果列表
        """
        return [self.query(q) for q in questions]

    @staticmethod
    def format_result(result: Dict[str, Any]) -> str:
        """格式化查询结果为可读文本。"""
        output = f"【问题】{result['question']}\n\n"
        output += f"【回答】\n{result['answer']}\n\n"
        output += f"【引用来源】({result['retrieved_count']} 个文档块)\n"
        for i, src in enumerate(result["sources"], 1):
            output += f"  {i}. {src['source']}\n"
        output += f"\n【检索策略】{result['strategy']}"
        return output
