"""
性能计时脚本：分别测量冷启动耗时和每次问答耗时，验证优化效果。

用法：python test_perf.py
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main import RAGLearningAssistant


def main():
    assistant = RAGLearningAssistant()

    # 冷启动计时（模型加载 + 索引加载 + 文档分块 + BM25 建索引）
    t0 = time.time()
    assistant.initialize_system()
    assistant.build_knowledge_base()
    cold_time = time.time() - t0
    print(f"\n[冷启动耗时] {cold_time:.1f}s（模型/索引加载，仅启动一次）")

    # 三类问题：明确问题(跳过改写)、list类、模糊问题(触发改写)
    questions = [
        "什么是混合检索？为什么比单一向量检索好？",   # 明确 → 跳过改写
        "RAG 有哪些检索方式？",                        # list → 不改写，规则路由
        "RAG",                                          # 模糊 → 触发 LLM 改写
    ]

    for q in questions:
        print("\n" + "=" * 60)
        t0 = time.time()
        answer = assistant.ask_question(q, stream=False)
        dt = time.time() - t0
        print(f"\n[本次问答耗时] {dt:.1f}s")
        print(f"--- 回答 ---\n{answer}")
        print("=" * 60)


if __name__ == "__main__":
    main()
