"""
P4 Advanced RAG Pipeline - 配置管理

支持 OpenAI / 通义千问(DashScope) / Ollama 等 OpenAI 兼容 API
通过 .env 文件或环境变量配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "docs" / "sample_docs"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
FAISS_INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"

# ── LLM 配置 ──────────────────────────────────────────────
# 兼容 OpenAI / DashScope / Ollama 等
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# ── Embedding 配置 ─────────────────────────────────────────
# 本地 sentence-transformers 模型（无需 API Key）
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Reranker 配置 ─────────────────────────────────────────
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── 分块参数 ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── 检索参数 ──────────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))

# ── 确保数据目录存在 ──────────────────────────────────────
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
