"""
临时验证脚本：构建知识库并问一个问题，验证改造后的 RAG 全链路
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main import RAGLearningAssistant


def main():
    assistant = RAGLearningAssistant()
    assistant.initialize_system()
    assistant.build_knowledge_base()

    print("\n" + "=" * 60)
    question = "什么是混合检索？为什么比单一向量检索好？"
    print(f"测试问题: {question}")
    answer = assistant.ask_question(question, stream=False)
    print("\n" + "=" * 60)
    print("最终回答：")
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
