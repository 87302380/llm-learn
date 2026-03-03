"""
生成器 - 带引用溯源的回答生成

演示要点:
- Prompt 工程: 上下文注入 + 引用指令
- 流式输出
- 引用溯源解析
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from document_loader import Document


# ── Prompt 模板 ───────────────────────────────────────────

RAG_SYSTEM_PROMPT = """你是一个专业的问答助手。请严格根据提供的参考文档来回答用户的问题。

## 回答规则:
1. 只使用参考文档中的信息回答问题
2. 如果参考文档中没有相关信息，请明确说明"根据提供的文档，无法回答该问题"
3. 在回答中标注引用来源，格式为 [来源X]
4. 回答要简洁、准确、有条理

## 参考文档:
{context}
"""

RAG_USER_PROMPT = """问题: {question}

请基于上述参考文档回答问题，并标注引用来源。"""


def format_context(documents: List[Tuple[Document, float]]) -> str:
    """
    格式化检索结果为上下文字符串

    每个文档块标注来源编号和元数据
    """
    context_parts = []
    for i, (doc, score) in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", "")
        section = doc.metadata.get("section_title", "")

        # 构建来源标签
        source_info = f"来源{i}: {source}"
        if page:
            source_info += f" (第{page}页)"
        if section:
            source_info += f" - {section}"

        context_parts.append(
            f"--- [{source_info}] (相关度: {score:.2f}) ---\n{doc.content}"
        )

    return "\n\n".join(context_parts)


class RAGGenerator:
    """RAG 回答生成器"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
    ):
        from openai import OpenAI
        from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

        self.model = model or LLM_MODEL
        self.client = OpenAI(
            api_key=api_key or LLM_API_KEY,
            base_url=base_url or LLM_BASE_URL,
        )
        self.temperature = temperature

    def generate(
        self,
        question: str,
        retrieved_docs: List[Tuple[Document, float]],
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """
        基于检索结果生成回答

        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档列表 (Document, score)
            chat_history: 可选的对话历史

        Returns:
            生成的回答文本
        """
        context = format_context(retrieved_docs)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
        ]

        # 加入对话历史（如果有）
        if chat_history:
            messages.extend(chat_history)

        messages.append(
            {"role": "user", "content": RAG_USER_PROMPT.format(question=question)}
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    def generate_stream(
        self,
        question: str,
        retrieved_docs: List[Tuple[Document, float]],
    ):
        """流式生成回答"""
        context = format_context(retrieved_docs)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": RAG_USER_PROMPT.format(question=question)},
        ]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
