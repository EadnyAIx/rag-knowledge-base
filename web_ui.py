"""RAG 知识库问答系统 - Streamlit Web UI。

运行方式: streamlit run web_ui.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from config import Config
from src import DocumentLoader, TextChunker, VectorStoreManager, Retriever, RAGChain


st.set_page_config(
    page_title="RAG 知识库问答系统",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_app():
    """初始化 RAG 应用（缓存）。"""
    Config.validate()
    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    vector_store = VectorStoreManager()
    retriever = Retriever(vector_store)
    rag_chain = RAGChain(retriever)
    return loader, chunker, vector_store, retriever, rag_chain


def main():
    st.title("📚 RAG 知识库问答系统")
    st.caption("基于 LangChain + Chroma + OpenAI 的检索增强生成系统")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统配置")
        st.info(f"向量库文档数: **{get_app()[2].count()}**")

        strategy = st.selectbox(
            "检索策略",
            options=["mmr", "similarity", "ensemble"],
            index=0,
            help="MMR: 兼顾相关性与多样性 | Similarity: 纯相似度 | Ensemble: 向量+关键词混合",
        )

        top_k = st.slider("Top-K 检索数量", min_value=1, max_value=10, value=Config.TOP_K)

        st.divider()
        st.header("📤 文档上传")
        uploaded_files = st.file_uploader(
            "上传 PDF/TXT/MD 文件",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("开始索引", type="primary"):
            with st.spinner("正在处理文档..."):
                loader, chunker, vector_store, _, _ = get_app()
                total = 0
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        docs = loader.load_file(tmp_path)
                        chunks = chunker.split_documents(docs)
                        ids = vector_store.add_documents(chunks)
                        total += len(ids)
                        st.success(f"✅ {uploaded_file.name}: 索引了 {len(ids)} 个块")
                    finally:
                        os.unlink(tmp_path)

                st.success(f"🎉 总共索引了 {total} 个文档块")
                st.rerun()

        if st.button("清空向量库"):
            if st.session_state.get("confirm_clear"):
                get_app()[2].delete()
                st.success("向量库已清空")
                st.session_state["confirm_clear"] = False
                st.rerun()
            else:
                st.session_state["confirm_clear"] = True
                st.warning("再次点击确认清空")

    # 主区域 - 问答
    st.header("💬 知识库问答")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 引用来源"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {src['source']}**")
                        st.text(src["content_preview"][:150] + "...")

    # 用户输入
    if question := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索和生成..."):
                _, _, _, retriever, rag_chain = get_app()
                result = rag_chain.query(question, strategy=strategy)
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📎 引用来源"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**{i}. {src['source']}**")
                            st.text(src["content_preview"][:150] + "...")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )


if __name__ == "__main__":
    main()
