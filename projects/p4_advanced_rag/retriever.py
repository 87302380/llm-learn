"""
检索器 - 混合检索 + HyDE + 重排序

演示要点:
- 密集检索 (Dense Retrieval) - 向量相似度
- 稀疏检索 (Sparse Retrieval) - BM25
- 混合检索 (Hybrid Retrieval) - 加权融合
- HyDE (Hypothetical Document Embedding)
- Cross-Encoder 重排序
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from document_loader import Document
from embedder import ChromaStore, LocalEmbedder


class BM25Retriever:
    """
    BM25 稀疏检索

    原理: 基于词频(TF) 和逆文档频率(IDF) 的经典检索算法
    优点: 对精确关键词匹配效果好，无需 GPU
    缺点: 无法理解语义相似性
    """

    def __init__(self):
        self._index = None
        self._documents: List[Document] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词（中英文混合）"""
        import re
        # 中文按字分词 + 英文按词分词
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
        return tokens

    def add_documents(self, documents: List[Document]):
        """构建 BM25 索引"""
        from rank_bm25 import BM25Okapi

        self._documents = documents
        tokenized_corpus = [self._tokenize(doc.content) for doc in documents]
        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """BM25 检索"""
        if self._index is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)

        # 按分数排序
        scored_docs = list(zip(self._documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [(doc, float(score)) for doc, score in scored_docs[:top_k]]


class HybridRetriever:
    """
    混合检索器 - 融合 Dense + Sparse

    融合策略: 加权 Reciprocal Rank Fusion (RRF)

    RRF 公式: score(d) = Σ 1/(k + rank_i(d))
    其中 k 是常数（通常取 60），rank_i(d) 是文档 d 在第 i 个检索器中的排名
    """

    def __init__(
        self,
        vector_store: ChromaStore,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        混合检索 - RRF 融合

        步骤:
        1. 分别用 Dense 和 Sparse 检索
        2. 对两路结果做 RRF 融合
        3. 按融合分数排序返回
        """
        # 1. Dense 检索
        dense_results = self.vector_store.search(query, top_k=top_k)
        # 2. Sparse 检索
        sparse_results = self.bm25_retriever.search(query, top_k=top_k)

        # 3. RRF 融合
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (doc, _score) in enumerate(dense_results):
            key = doc.content[:100]  # 用内容前100字作为去重 key
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        for rank, (doc, _score) in enumerate(sparse_results):
            key = doc.content[:100]
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        # 排序
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [(doc_map[key], score) for key, score in sorted_results[:top_k]]


class HyDERetriever:
    """
    HyDE (Hypothetical Document Embedding)

    原理:
    1. 用 LLM 根据查询生成一个"假设性回答"
    2. 用这个假设性回答去做向量检索
    3. 假设性回答与真实文档在向量空间中更接近

    论文: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
    """

    def __init__(
        self,
        vector_store: ChromaStore,
        llm_client=None,
        llm_model: str = "gpt-3.5-turbo",
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.llm_model = llm_model

    def _generate_hypothetical_doc(self, query: str) -> str:
        """用 LLM 生成假设性文档"""
        if self.llm_client is None:
            from openai import OpenAI
            from config import LLM_API_KEY, LLM_BASE_URL
            self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "请根据问题写一段简短的回答（约100字），作为假设性文档用于检索。"},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """HyDE 检索"""
        # 1. 生成假设性文档
        hypo_doc = self._generate_hypothetical_doc(query)
        print(f"  📝 HyDE 假设文档: {hypo_doc[:80]}...")

        # 2. 用假设性文档做向量检索
        return self.vector_store.search(hypo_doc, top_k=top_k)


class CrossEncoderReranker:
    """
    Cross-Encoder 重排序

    原理:
    - Bi-Encoder: query 和 doc 分别编码，再算相似度（快但不精确）
    - Cross-Encoder: query 和 doc 拼接后一起编码（慢但精确）

    使用场景: 先用 Bi-Encoder 粗召回，再用 Cross-Encoder 精排
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """
        重排序

        Args:
            query: 查询文本
            documents: (Document, score) 列表
            top_k: 返回前 K 个

        Returns:
            重排序后的 (Document, score) 列表
        """
        if not documents:
            return []

        model = self._get_model()

        # 构建 query-document 对
        pairs = [(query, doc.content) for doc, _ in documents]

        # Cross-Encoder 打分
        scores = model.predict(pairs)

        # 按新分数排序
        reranked = list(zip([doc for doc, _ in documents], scores))
        reranked.sort(key=lambda x: x[1], reverse=True)

        return [(doc, float(score)) for doc, score in reranked[:top_k]]


if __name__ == "__main__":
    from document_loader import Document

    # 测试数据
    test_docs = [
        Document(content="大语言模型（LLM）是一种基于 Transformer 架构的深度学习模型，通过在大规模语料上预训练来学习语言规律。", metadata={"source": "llm_intro"}),
        Document(content="RAG（Retrieval-Augmented Generation）检索增强生成，是将信息检索与文本生成相结合的技术方案。", metadata={"source": "rag_intro"}),
        Document(content="BM25 是一种基于概率的经典信息检索算法，它考虑了词频、文档长度和逆文档频率。", metadata={"source": "bm25_intro"}),
        Document(content="向量数据库如 ChromaDB、FAISS、Milvus，专门用于高维向量的存储和近似最近邻检索。", metadata={"source": "vector_db"}),
        Document(content="Python 的 FastAPI 框架用于构建高性能的 REST API，支持类型注解和自动文档生成。", metadata={"source": "fastapi"}),
    ]

    # 测试 BM25
    print("🔍 BM25 检索测试:")
    bm25 = BM25Retriever()
    bm25.add_documents(test_docs)
    results = bm25.search("什么是 RAG 检索增强", top_k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.content[:50]}...")

    # 测试混合检索
    print("\n🔍 混合检索测试:")
    embedder = LocalEmbedder()
    chroma = ChromaStore(embedder=embedder)
    chroma.add_documents(test_docs)

    hybrid = HybridRetriever(chroma, bm25)
    results = hybrid.search("什么是 RAG 检索增强", top_k=3)
    for doc, score in results:
        print(f"  [{score:.6f}] {doc.content[:50]}...")
