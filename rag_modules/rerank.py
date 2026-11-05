"""
重排序模块
"""

import numpy as np
from typing import List, Optional

from langchain_core.documents import Document


class EmbeddingReranker:
    """基于嵌入相似度的重排器，对混合检索候选集做二次精排。

    混合检索里的 RRF 融合只做「排名合并」：它只看每个文档在两路检索里
    排第几，不重新读文档内容和 query 的语义匹配度。本重排器用嵌入模型
    把 query 和每个候选文档重新编码、算余弦相似度，按相似度从高到低
    精排，弥补 RRF「只依赖排名、不依赖内容」的不足。

    设计说明：
    - 复用索引构建时的同一个嵌入模型，不额外加载模型、不额外占显存/内存。
    - 只在「召回阶段之后」对候选集精排（宽进严出），不改变召回逻辑。
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings

    @staticmethod
    def _cosine(a, b) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def rerank(self, query: str, documents: List[Document], top_n: Optional[int] = None) -> List[Document]:
        """
        对候选文档按与 query 的余弦相似度重排。

        Args:
            query: 查询文本
            documents: 待重排的候选文档列表
            top_n: 返回前 N 条，None 表示返回全部（按相似度降序）

        Returns:
            重排后的文档列表
        """
        if not documents:
            return documents

        q_vec = self.embeddings.embed_query(query)
        doc_vecs = self.embeddings.embed_documents([d.page_content for d in documents])

        scored = [(self._cosine(q_vec, dv), d) for dv, d in zip(doc_vecs, documents)]
        scored.sort(key=lambda x: x[0], reverse=True)

        if top_n:
            return [d for _, d in scored[:top_n]]
        return [d for _, d in scored]
