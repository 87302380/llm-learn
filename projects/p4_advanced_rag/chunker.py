"""
文档分块器 - 多种分块策略对比

演示要点:
- 固定大小分块
- 递归字符分块（LangChain 风格）
- 语义分块（基于 Embedding 相似度）
"""

from __future__ import annotations

import re
from typing import List, Literal

from document_loader import Document


class FixedSizeChunker:
    """
    固定大小分块 - 最简单的策略

    优点: 简单可控
    缺点: 可能在句子中间截断
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text = doc.content
            start = 0
            chunk_index = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(Document(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_index,
                            "chunk_strategy": "fixed_size",
                        }
                    ))
                    chunk_index += 1
                start = end - self.chunk_overlap
        return chunks


class RecursiveChunker:
    """
    递归字符分块 - LangChain 默认策略

    按照分隔符优先级递归拆分:
    1. 双换行 (段落)
    2. 单换行
    3. 句号/问号/感叹号
    4. 空格
    5. 字符级别
    """

    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归拆分文本"""
        final_chunks: List[str] = []

        # 找到第一个能拆分的分隔符
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # 拆分
        splits = text.split(separator) if separator else list(text)

        # 合并短片段
        current_chunk = ""
        for split_text in splits:
            piece = split_text if not separator else split_text + separator
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                # 如果单个片段太长，用下一个分隔符继续拆
                if len(piece) > self.chunk_size and separators.index(separator) < len(separators) - 1:
                    remaining_seps = separators[separators.index(separator) + 1:]
                    sub_chunks = self._split_text(piece, remaining_seps)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece

        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        return final_chunks

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text_chunks = self._split_text(doc.content, self.SEPARATORS)
            for i, chunk_text in enumerate(text_chunks):
                if chunk_text:
                    chunks.append(Document(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "chunk_strategy": "recursive",
                        }
                    ))
        return chunks


class SemanticChunker:
    """
    语义分块 - 基于 Embedding 相似度

    原理: 先按句子拆分，计算相邻句子的 Embedding 余弦相似度，
    在相似度突变处（低于阈值）切分 chunk。

    优点: chunk 内语义连贯性强
    缺点: 需要额外的 Embedding 计算
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """按句子分割（中英文混合）"""
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a, b) -> float:
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def split(self, documents: List[Document]) -> List[Document]:
        model = self._get_model()
        chunks = []

        for doc in documents:
            sentences = self._split_sentences(doc.content)
            if len(sentences) <= 1:
                chunks.append(Document(
                    content=doc.content,
                    metadata={**doc.metadata, "chunk_index": 0, "chunk_strategy": "semantic"}
                ))
                continue

            # 计算句子 embeddings
            embeddings = model.encode(sentences)

            # 找到语义断点
            current_chunk_sentences = [sentences[0]]
            chunk_index = 0

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                current_text = " ".join(current_chunk_sentences)

                if sim < self.similarity_threshold and len(current_text) >= self.min_chunk_size:
                    # 语义断点 → 创建新 chunk
                    chunks.append(Document(
                        content=current_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_index,
                            "chunk_strategy": "semantic",
                        }
                    ))
                    chunk_index += 1
                    current_chunk_sentences = [sentences[i]]
                else:
                    current_chunk_sentences.append(sentences[i])

            # 最后一个 chunk
            if current_chunk_sentences:
                chunks.append(Document(
                    content=" ".join(current_chunk_sentences),
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_index,
                        "chunk_strategy": "semantic",
                    }
                ))

        return chunks


# ── 分块策略注册表 ─────────────────────────────────────────
ChunkStrategy = Literal["fixed", "recursive", "semantic"]

CHUNKERS = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def create_chunker(
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs,
):
    """工厂方法: 创建分块器"""
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    return CHUNKERS[strategy](chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if __name__ == "__main__":
    # 快速测试
    test_docs = [Document(
        content="""大语言模型（LLM）是一种基于 Transformer 架构的深度学习模型。它通过在大规模文本语料上进行预训练来学习语言的统计规律。

GPT 系列由 OpenAI 开发，是目前最知名的大语言模型之一。GPT-4 具备多模态理解能力，可以同时处理文本和图像输入。

RAG（检索增强生成）是一种结合信息检索与生成模型的技术。它首先从知识库中检索相关文档，然后将检索结果作为上下文提供给生成模型。""",
        metadata={"source": "test.txt"}
    )]

    for strategy in ["fixed", "recursive"]:
        chunker = create_chunker(strategy, chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(test_docs)
        print(f"\n{'='*60}")
        print(f"策略: {strategy} | 分块数: {len(chunks)}")
        for c in chunks:
            print(f"  [{c.metadata['chunk_index']}] ({len(c.content)} chars) {c.content[:60]}...")
