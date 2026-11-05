"""
RAG 学习问答助手 —— Streamlit 界面

把 RAG 的完整处理链路（路由 → 改写 → 过滤 → 混合检索 → RRF → Rerank → 父子文档 → 生成）
可视化出来，让"检索"不再是一个黑盒。

启动：streamlit run app.py
"""

import sys
from pathlib import Path

# 保证能 import 同目录的 main.py
sys.path.append(str(Path(__file__).parent))

import streamlit as st

from main import RAGLearningAssistant

# 页面配置
st.set_page_config(
    page_title="RAG 学习问答助手",
    page_icon="📚",
    layout="wide",
)

ROUTE_LABELS = {
    "list": "📋 盘点列举",
    "detail": "🎯 深挖概念",
    "general": "💬 一般问答",
}


@st.cache_resource(show_spinner=False)
def get_assistant() -> RAGLearningAssistant:
    """加载助手实例（缓存后模型常驻内存，冷启动只发生一次）"""
    assistant = RAGLearningAssistant()
    assistant.initialize_system()
    assistant.build_knowledge_base()
    return assistant


def render_trace(trace: dict):
    """渲染一次问答的检索过程 trace"""
    route_type = trace.get("route_type", "general")
    route_label = ROUTE_LABELS.get(route_type, route_type)

    c1, c2, c3 = st.columns(3)
    c1.metric("路由类型", route_label)
    c2.metric("查询改写", "✍️ 是" if trace.get("did_rewrite") else "⏭️ 否（问题已明确）")
    c3.metric("检索耗时", f"{trace.get('elapsed', 0)}s")

    if trace.get("did_rewrite"):
        st.write(f"✍️ 改写后查询：`{trace.get('rewritten_query')}`")

    if trace.get("filters"):
        st.write(f"🎯 章节过滤：`{trace.get('filters')}`")
    else:
        st.write("🎯 章节过滤：无（全局检索）")

    chunks = trace.get("chunks", [])
    if chunks:
        st.markdown(f"**📄 检索到的文档块（精排后 Top {len(chunks)}）：**")
        for i, c in enumerate(chunks, 1):
            chapter = f" ｜ {c['chapter']}" if c.get("chapter") else ""
            st.markdown(f"**{i}. {c['doc_title']}** · {c['section']}{chapter}")
            st.caption(c["preview"] + " …")
            st.divider()
    else:
        st.write("⚠️ 未检索到相关内容")

    doc_names = trace.get("doc_names", [])
    if doc_names:
        st.write(f"📚 命中的完整文档：{', '.join(doc_names)}")


def main():
    st.title("📚 RAG 学习问答助手")
    st.caption("基于 all-in-rag 教程文档构建 · 完整展示「检索增强生成」每一步")

    # 加载助手（首次冷启动约 15 秒，之后模型常驻内存）
    with st.spinner("正在加载知识库与模型（首次约 15 秒，请稍候）..."):
        assistant = get_assistant()

    # ===== 侧边栏 =====
    with st.sidebar:
        st.header("⚙️ 系统信息")
        stats = assistant.data_module.get_statistics()
        m1, m2 = st.columns(2)
        m1.metric("教程文档", f"{stats['total_documents']} 篇")
        m2.metric("文本块", f"{stats['total_chunks']} 块")

        with st.expander("📂 章节分布", expanded=False):
            for chapter, count in stats["chapters"].items():
                st.text(f"{chapter}：{count} 块")

        st.divider()
        st.subheader("🔗 处理链路")
        st.markdown(
            """
            1. **查询路由** — 判断问题类型
            2. **查询改写** — 模糊问题补全术语
            3. **章节过滤** — 锁定相关章节
            4. **混合检索** — 稠密向量 + BM25
            5. **RRF 融合** — 合并两路结果
            6. **Rerank 精排** — 二次打分
            7. **父子文档** — 小块召回完整文档
            8. **生成回答** — 基于检索内容作答
            """
        )

        st.divider()
        st.subheader("💡 试试这些示例")
        examples = [
            "什么是混合检索？为什么比单一向量检索好？",
            "RAG 有哪些检索方式？",
            "什么是 Rerank 重排序？",
            "向量数据库和传统数据库有什么区别？",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.pending_prompt = ex

        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []

        st.divider()
        st.caption(
            f"检索 Top-K = {assistant.config.top_k} · 生成模型 = {assistant.config.llm_model} · 嵌入 = bge-small-zh"
        )

    # ===== 主区域：聊天 =====
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                with st.expander("🔍 查看本次检索过程", expanded=False):
                    render_trace(msg["trace"])

    # 处理输入
    prompt = st.chat_input("问我一个 RAG 相关问题，例如：什么是混合检索？")
    if not prompt and st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 助手回答
        with st.chat_message("assistant"):
            with st.status("检索中 → 生成中...", expanded=False) as status:
                answer, trace = assistant.ask_question(prompt, stream=False, return_trace=True)
                status.update(label="✅ 完成", state="complete", expanded=False)

            st.markdown(answer)

            with st.expander("🔍 查看本次检索过程", expanded=True):
                render_trace(trace)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "trace": trace}
        )


if __name__ == "__main__":
    main()
