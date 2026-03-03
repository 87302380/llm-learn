"""
文档加载器 - 支持 PDF / Markdown / TXT 多格式解析

演示要点:
- 统一的文档抽象 (Document dataclass)
- 多格式解析策略模式
- 元数据提取
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    """统一的文档数据结构"""
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(source={self.source!r}, len={len(self.content)}, preview={preview!r}...)"


class TextLoader:
    """纯文本加载器"""

    def load(self, file_path: Path) -> List[Document]:
        text = file_path.read_text(encoding="utf-8")
        return [Document(
            content=text,
            metadata={"source": str(file_path), "format": "txt"}
        )]


class MarkdownLoader:
    """Markdown 加载器 - 按标题分节"""

    def load(self, file_path: Path) -> List[Document]:
        text = file_path.read_text(encoding="utf-8")
        # 按一级/二级标题分节
        sections = re.split(r'\n(?=#{1,2}\s)', text)
        docs = []
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            # 提取标题
            title_match = re.match(r'^(#{1,2})\s+(.+)', section)
            title = title_match.group(2) if title_match else f"Section {i}"
            docs.append(Document(
                content=section,
                metadata={
                    "source": str(file_path),
                    "format": "markdown",
                    "section_title": title,
                    "section_index": i,
                }
            ))
        return docs if docs else [Document(
            content=text,
            metadata={"source": str(file_path), "format": "markdown"}
        )]


class PDFLoader:
    """PDF 加载器 - 逐页提取文本"""

    def load(self, file_path: Path) -> List[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")

        reader = PdfReader(str(file_path))
        docs = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                docs.append(Document(
                    content=text,
                    metadata={
                        "source": str(file_path),
                        "format": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": len(reader.pages),
                    }
                ))
        return docs


# ── 加载器注册表 ──────────────────────────────────────────
LOADERS = {
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".pdf": PDFLoader,
}


def load_documents(path: Path) -> List[Document]:
    """
    智能加载文档 - 自动识别格式

    Args:
        path: 文件或目录路径

    Returns:
        文档列表
    """
    if path.is_file():
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            print(f"⚠️ 不支持的文件格式: {path.suffix}, 使用纯文本加载")
            loader_cls = TextLoader
        return loader_cls().load(path)

    elif path.is_dir():
        all_docs = []
        for ext, loader_cls in LOADERS.items():
            for file_path in sorted(path.rglob(f"*{ext}")):
                try:
                    docs = loader_cls().load(file_path)
                    all_docs.extend(docs)
                    print(f"  ✅ 已加载: {file_path.name} ({len(docs)} 个段落)")
                except Exception as e:
                    print(f"  ❌ 加载失败: {file_path.name} - {e}")
        return all_docs
    else:
        raise FileNotFoundError(f"路径不存在: {path}")


if __name__ == "__main__":
    from config import DOCS_DIR
    print(f"📂 从 {DOCS_DIR} 加载文档...")
    docs = load_documents(DOCS_DIR)
    print(f"\n📊 共加载 {len(docs)} 个文档段落")
    for doc in docs[:3]:
        print(f"  - {doc}")
