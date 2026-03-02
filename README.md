# LLM 工程师学习路径与 Colab 实战

本仓库提供可直接在 Colab 运行的 Notebook：

- `colab/llm_engineer_learning_path_projects.ipynb`
- `colab/phase4_rag_agent_projects.ipynb`

内容包括：
- 从传统 NLP 转向大模型工程的阶段化学习路径
- 项目 1：AG News 文本分类微调（Transformers Trainer）
- 项目 2：SQuAD 向量检索（SentenceTransformer + FAISS）
- 项目 3：RAG 问答（检索 + Prompt 注入 + FLAN-T5 生成）

新增阶段四核心实践 Notebook：
- P10：RAG 全链路（解析、向量化、检索、生成）
- P11：Advanced RAG（BM25/混合召回、HyDE、重排序）
- P12：Agent 与工具调用（ReAct/Tool Use）

## Colab 使用方式
1. 打开 Colab，上传该 Notebook。
2. 选择 GPU 运行时（推荐 T4）。
3. 顺序执行每个代码块。

## 数据集来源
- `ag_news`（Hugging Face Datasets）
- `squad`（Hugging Face Datasets）
