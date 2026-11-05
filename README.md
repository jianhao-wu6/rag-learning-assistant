# RAG 学习问答助手

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

基于 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 教程文档（36 篇、423 个文本块）构建的 **RAG（检索增强生成）学习问答系统**，把 RAG 的完整优化链路落地为一个可交互、可追溯的作品。

> 本项目是 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)（Datawhale 出品，CC BY-NC-SA 4.0）`code/C8` 章节代码的**衍生作品**，由原「尝尝咸淡」食谱 RAG 系统改造而来，用于实践并展示 RAG 的核心技术。依据上游「相同方式共享」要求，本项目同样采用 **CC BY-NC-SA 4.0** 许可，详见 [LICENSE](./LICENSE)。

---

## ✨ 核心特性

- **完整 RAG 优化链**：`查询路由 → 查询改写 → 章节过滤 → 混合检索 → RRF 融合 → Rerank 精排 → 父子文档召回 → 生成`
- **混合检索**：稠密向量（`bge-small-zh-v1.5`）+ 稀疏检索（BM25），RRF 倒数排名融合
- **Rerank 重排序**：双塔嵌入模型对候选集二次精排，「宽进严出」（召回 20 → 精排 3）
- **父子文档检索**：小块精准召回，完整文档作为上下文作答，兼顾精度与完整度
- **查询路由 + 改写**：规则路由省延迟，模糊问题才触发 LLM 改写
- **全过程可视化**：Streamlit 界面实时展示路由结果、章节过滤、检索到的文档块与耗时

---

## 🏗️ 技术架构

```
用户提问
   │
   ├─① 查询路由（规则判断：list / detail / general）
   ├─② 查询改写（模糊问题才调用 LLM，明确问题跳过）
   ├─③ 章节过滤（关键词→章节映射，最长关键词优先）
   ├─④ 混合检索（稠密向量 FAISS + 稀疏 BM25）
   ├─⑤ RRF 融合（倒数排名融合，只看排名不看分数）
   ├─⑥ Rerank 精排（嵌入模型二次打分，宽进严出）
   ├─⑦ 父子文档（小块 → 完整父文档）
   └─⑧ 生成（DeepSeek，基于检索内容作答，不足则如实说明）
```

## 📁 目录结构

```
code/C8/
├── main.py                        # 主程序：RAGLearningAssistant
├── app.py                         # Streamlit 界面
├── config.py                      # 配置（数据路径、模型、top_k）
├── requirements.txt               # 依赖
├── test_rag.py                    # 全链路验证脚本
├── test_perf.py                   # 性能计时脚本
└── rag_modules/
    ├── data_preparation.py        # 数据加载 + Markdown 结构分块 + 元数据提取
    ├── index_construction.py      # FAISS 向量索引构建 / 加载
    ├── retrieval_optimization.py  # 混合检索 + RRF + 元数据过滤
    ├── rerank.py                  # Rerank 重排序（EmbeddingReranker）
    └── generation_integration.py  # DeepSeek 接入 + 路由/改写/生成 prompt
```

## 🚀 快速开始

### 0. 获取知识库（前置）

本项目知识库指向 `data/` 目录（见 `config.py` 的 `data_path`）。运行前请把教程文档（`.md`）放入 `data/` 目录：

```bash
# 克隆上游 all-in-rag，获取教程文档
git clone https://github.com/datawhalechina/all-in-rag.git
# 把教程文档复制到本项目的 data/ 目录
cp -r all-in-rag/docs/* data/
```

> 系统会自动递归扫描 `data/` 下所有 `.md` 文件；也可修改 `config.py` 的 `data_path` 指向你自己的知识库目录，详见 `data/README.md`。

### 1. 环境准备

```bash
conda create -n all-in-rag python=3.11
conda activate all-in-rag
pip install -r requirements.txt
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，并填入 DeepSeek API Key：

```bash
cp .env.example .env
```

`.env` 内容：

```
DEEPSEEK_API_KEY=sk-xxx
```

### 3. 启动（二选一）

**命令行交互**（适合调试）：

```bash
python main.py
```

**Streamlit 界面**（作品展示）：

```bash
streamlit run app.py
```

首次启动会加载嵌入模型 + 构建向量索引（约 15 秒），之后模型常驻内存，问答秒级响应。

---

## 🎯 技术亮点与取舍

| 环节 | 方案 | 为什么 |
|---|---|---|
| 混合检索 | 稠密 + 稀疏 + RRF | 稠密管语义泛化（"番茄"≈"西红柿"），稀疏管精确匹配（型号、函数名），RRF 只看排名天然免疫分数尺度差异 |
| Rerank | 双塔嵌入精排 | RRF 只看排名不重看内容，精排把排序不准的纠偏回来；宽进严出避免召回就漏 |
| 父子文档 | 小块召回 + 完整文档作答 | 小块匹配更准，完整文档提供充分上下文，减少断章取义 |
| 查询路由 | 规则路由（替代 LLM） | 学习问答的路由信号明确，规则更稳、零延迟、省 token |
| 查询改写 | 明确跳过、模糊才改写 | 多数问题无需改写，省一次 LLM 调用；模糊问题补全术语提升召回 |
| 元数据过滤 | 章节映射 + 最长关键词优先 | 先锁定章节再检索，缩小候选空间、减少噪声 |

## 📌 诚实说明

- 生成环节是耗时大头（约 8–9s），检索全链路仅约 1s；「检索」本身不是延迟瓶颈。
- Rerank 复用 bge-small 双塔打分，效果弱于专用 cross-encoder 重排器，但零额外模型开销，性价比高。
- 教程知识库共 36 篇，覆盖面有限；换知识库只需替换 `docs/` 目录与 `config.py` 的 `data_path`。

---

## 📄 许可证

本项目是 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)（Datawhale 出品）的衍生作品，依据其「相同方式共享」要求，采用与上游一致的 **CC BY-NC-SA 4.0**（知识共享署名—非商业性使用—相同方式共享 4.0 国际许可协议）。

- 署名：请注明上游作者 Datawhale / all-in-rag，并提供许可协议链接。
- 非商业：不得用于商业用途。
- 相同方式共享：基于本项目的衍生作品须以相同协议分发。

详见 [LICENSE](./LICENSE)。
