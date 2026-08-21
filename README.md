# 📚 RAG 知识库问答系统

> 基于 LangChain + ChromaDB + OpenAI 的检索增强生成（RAG）系统，支持多格式文档索引、多种检索策略和引用溯源。

## ✨ 功能特性

- **多格式文档加载**: 支持 PDF、TXT、Markdown 等格式
- **智能文本分块**: 递归字符分块，支持自定义块大小和重叠
- **向量持久化存储**: 基于 ChromaDB，数据持久化到本地
- **多种检索策略**:
  - 相似度检索 (Similarity Search)
  - 最大边际相关性 (MMR)
  - 混合检索 (Ensemble: 向量 + BM25)
- **引用溯源**: 回答附带来源文档信息
- **流式输出**: 支持逐 token 流式生成
- **Web 界面**: 基于 Streamlit 的交互式 UI
- **CLI 工具**: 命令行索引和问答

## 🏗️ 系统架构

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  文档输入    │────▶│  文档加载器   │────▶│  文本分块器   │
│ (PDF/TXT/MD)│     │              │     │              │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  用户提问    │────▶│   检索器      │◀────│  向量存储     │
│             │     │ (3种策略)     │     │  (ChromaDB)  │
└──────┬──────┘     └──────┬───────┘     └──────────────┘
       │                     │
       ▼                     ▼
┌─────────────┐     ┌──────────────┐
│  LLM 生成    │◀────│  上下文组装   │
│ (OpenAI)    │     │              │
└──────┬──────┘     └──────────────┘
       │
       ▼
┌─────────────┐
│  回答+来源   │
└─────────────┘
```

## 📦 安装

```bash
# 克隆仓库
git clone <repo-url>
cd rag-knowledge-base

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY
```

## 🚀 使用方法

### 1. 索引文档

```bash
# 索引单个文件
python main.py index path/to/document.pdf

# 索引整个目录
python main.py index path/to/documents/
```

### 2. 交互式问答

```bash
python main.py chat
```

### 3. 单次查询

```bash
python main.py query "你的问题是什么？"
```

### 4. 查看系统状态

```bash
python main.py stats
```

### 5. 启动 Web UI

```bash
python main.py web
# 或直接运行
streamlit run web_ui.py
```

## 📁 项目结构

```
rag-knowledge-base/
├── main.py                 # 主程序入口（CLI）
├── web_ui.py               # Streamlit Web 界面
├── config.py               # 全局配置管理
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # 文档加载器（多格式支持）
│   ├── text_splitter.py    # 文本分块器
│   ├── vector_store.py     # 向量存储管理器（ChromaDB）
│   ├── retriever.py        # 检索器（3种策略）
│   └── rag_chain.py        # RAG 问答链
└── examples/
    └── sample_docs/
        └── example.md      # 示例文档
```

## 🔧 配置说明

在 `.env` 文件中配置：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | 必填 |
| `OPENAI_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-3-small` |
| `CHAT_MODEL` | 对话模型 | `gpt-4o-mini` |
| `CHUNK_SIZE` | 文本分块大小 | `500` |
| `CHUNK_OVERLAP` | 分块重叠大小 | `50` |
| `TOP_K` | 检索返回数量 | `4` |
| `RETRIEVAL_STRATEGY` | 检索策略 | `mmr` |

## 🎯 检索策略对比

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| `similarity` | 向量余弦相似度 | 语义匹配为主的查询 |
| `mmr` | 最大边际相关性，兼顾相关性与多样性 | 需要避免重复内容的查询 |
| `ensemble` | 向量检索 + BM25 关键词混合 | 需要同时匹配语义和精确关键词 |

## 📝 核心代码亮点

1. **模块化设计**: 每个功能独立成类，便于扩展和测试
2. **策略模式**: 检索策略可动态切换，无需修改核心逻辑
3. **元数据追踪**: 每个文档块保留来源、分块索引等元数据
4. **LangChain 原生集成**: 兼容 LangChain 的 Retriever/Chain 生态

## 🤝 扩展方向

- [ ] 支持更多文档格式（Word、Excel、PPT）
- [ ] 添加文档解析 OCR 能力
- [ ] 实现重排序（Reranker）模块
- [ ] 添加对话历史记忆
- [ ] 支持多租户知识库隔离
- [ ] 添加评估指标（召回率、准确率）

## 📄 License

MIT
