"""
Embedding & 向量存储管理

演示要点:
- sentence-transformers 本地 Embedding
- ChromaDB 向量存储
- FAISS 向量存储
- 统一接口方便切换
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from document_loader import Document


class LocalEmbedder:
    """本地 Embedding 模型封装（sentence-transformers）"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本为向量"""
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """编码查询文本"""
        return self.embed([query])[0]


class ChromaStore:
    """
    ChromaDB 向量存储

    优点: 简单易用、自带持久化、支持元数据过滤
    适用于: 中小规模文档（< 100w 条）
    """

    def __init__(
        self,
        collection_name: str = "rag_docs",
        persist_dir: Optional[str] = None,
        embedder: Optional[LocalEmbedder] = None,
    ):
        import chromadb

        self.embedder = embedder or LocalEmbedder()

        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document]) -> int:
        """添加文档到向量库"""
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed(texts)
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [
            {k: str(v) for k, v in doc.metadata.items()}
            for doc in documents
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """向量相似度检索"""
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs_with_scores = []
        for i in range(len(results["documents"][0])):
            doc = Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            )
            # ChromaDB 返回的是距离，转换为相似度 (cosine distance → similarity)
            score = 1.0 - results["distances"][0][i]
            docs_with_scores.append((doc, score))

        return docs_with_scores

    @property
    def count(self) -> int:
        return self.collection.count()


class FAISSStore:
    """
    FAISS 向量存储

    优点: 高性能、支持大规模、Meta 开源
    适用于: 大规模向量检索（> 100w 条）
    """

    def __init__(self, embedder: Optional[LocalEmbedder] = None):
        self.embedder = embedder or LocalEmbedder()
        self._index = None
        self._documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> int:
        """添加文档到 FAISS 索引"""
        import numpy as np
        import faiss

        texts = [doc.content for doc in documents]
        embeddings = np.array(self.embedder.embed(texts), dtype="float32")

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # Inner Product (cosine after normalize)
            faiss.normalize_L2(embeddings)

        self._index.add(embeddings)
        self._documents.extend(documents)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """向量相似度检索"""
        import numpy as np
        import faiss

        query_embedding = np.array([self.embedder.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self._index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._documents) and idx >= 0:
                results.append((self._documents[idx], float(score)))
        return results

    def save(self, path: str):
        """持久化索引"""
        import faiss
        faiss.write_index(self._index, f"{path}/index.faiss")
        # 保存文档
        docs_data = [{"content": d.content, "metadata": d.metadata} for d in self._documents]
        with open(f"{path}/documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False)

    def load(self, path: str):
        """加载索引"""
        import faiss
        self._index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/documents.json", "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        self._documents = [Document(**d) for d in docs_data]

    @property
    def count(self) -> int:
        return len(self._documents)


if __name__ == "__main__":
    # 快速测试
    test_docs = [
        Document(content="大语言模型 LLM 是基于 Transformer 的深度学习模型", metadata={"source": "test"}),
        Document(content="RAG 是检索增强生成技术，结合检索与生成", metadata={"source": "test"}),
        Document(content="向量数据库用于存储和检索高维向量数据", metadata={"source": "test"}),
        Document(content="Python 是最流行的编程语言之一", metadata={"source": "test"}),
    ]

    print("🔧 测试 ChromaDB...")
    store = ChromaStore()
    store.add_documents(test_docs)
    results = store.search("什么是 RAG?", top_k=2)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.content[:60]}")
