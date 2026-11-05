"""
生成集成模块
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "deepseek-chat", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        # DeepSeek 提供 OpenAI 兼容接口，用 ChatOpenAI 指定 base_url 接入
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的 RAG（检索增强生成）学习助教。请根据下面检索到的教程资料，用通俗易懂的语言回答学习者的问题。

用户问题: {question}

检索到的相关教程内容:
{context}

请遵守以下原则：
- 优先依据上面给出的教程内容作答，不要凭空编造教程里没有的知识点
- 如果教程内容不足以完整回答，请诚实说明"这部分资料里没有详细展开"，并给出你的补充理解
- 回答要条理清晰，适合正在学习 RAG 的新手理解

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的 RAG（检索增强生成）学习助教。请根据检索到的教程资料，为学习者深入讲解某个概念或技术点。

用户问题: {question}

检索到的相关教程内容:
{context}

请围绕问题把概念讲透，建议包含以下部分（可根据实际内容调整）：

## 🎯 一句话理解
[用一句话概括这个概念的直观理解]

## 🧩 核心原理
[讲清楚它是什么、怎么工作，尽量用类比帮助理解]

## 🔧 在 RAG 中的作用 / 做法
[它在完整 RAG 流程中处于哪一步、解决什么问题、常见做法]

## 💡 易错点 / 补充
[仅当教程或你的理解中有值得提醒的点时包含，可省略]

注意：
- 严格以教程内容为准，不要编造教程里没有的细节
- 教程没讲到的点要如实说明，再补充自己的理解
- 语气亲切、结构化，适合自学

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析学习者的提问，判断是否需要重写，以提高从 RAG 教程文档中检索的准确性。

原始查询: {query}

分析规则：
1. **具体明确的问题**（直接返回原查询）：
   - 包含明确术语：如"混合检索的RRF怎么算"、"Milvus如何建索引"
   - 明确的流程询问：如"查询重写有哪几种方式"、"什么是父子文档检索"

2. **模糊、口语化、过于宽泛的问题**（需要重写）：
   - 口语化或省略：如"这个怎么用"、"那是什么"（缺少上下文指代）
   - 过于宽泛：如"RAG"、"向量数据库"、"检索"
   - 表述不清：如"怎么提升效果"、"这块没看懂"

重写原则：
- 保持原意不变
- 补全省略的术语，把口语换成规范的 RAG 术语
- 保持简洁，不要过度扩写

示例：
- "RAG" → "RAG 检索增强生成 是什么"
- "怎么提升效果" → "RAG 如何提升检索和生成效果"
- "RRF是啥" → "RRF 倒数排名融合 混合检索重排序"
- "混合检索怎么做" → "混合检索怎么做"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据学习者的提问，将其分类为以下三种类型之一：

1. 'list' - 学习者想"盘点/列举"一类内容（概念清单、方法、工具、步骤概览）
   例如：RAG 有哪些检索方式、文本分块有哪几种方法、常用向量数据库有哪些

2. 'detail' - 学习者想深入理解某个具体概念、流程或技术细节
   例如：混合检索怎么做、RRF 为什么能提升效果、Milvus 怎么建索引

3. 'general' - 其他一般性问题
   例如：RAG 是什么、为什么要用 RAG、RAG 和微调的区别

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'  # 默认类型

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的内容。"

        # 提取文档标题
        titles = []
        for doc in context_docs:
            title = doc.metadata.get('doc_title', '未知主题')
            if title not in titles:
                titles.append(title)

        # 构建简洁的列表回答
        if len(titles) == 1:
            return f"相关内容：{titles[0]}"
        elif len(titles) <= 3:
            return f"为你梳理出以下内容：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(titles)])
        else:
            return f"为你梳理出以下内容：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(titles[:3])]) + f"\n\n还有其他 {len(titles)-3} 个相关主题。"

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的 RAG（检索增强生成）学习助教。请根据下面检索到的教程资料，用通俗易懂的语言回答学习者的问题。

用户问题: {question}

检索到的相关教程内容:
{context}

请遵守以下原则：
- 优先依据上面给出的教程内容作答，不要凭空编造教程里没有的知识点
- 如果教程内容不足以完整回答，请诚实说明"这部分资料里没有详细展开"，并给出你的补充理解
- 回答要条理清晰，适合正在学习 RAG 的新手理解

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的 RAG（检索增强生成）学习助教。请根据检索到的教程资料，为学习者深入讲解某个概念或技术点。

用户问题: {question}

检索到的相关教程内容:
{context}

请围绕问题把概念讲透，建议包含以下部分（可根据实际内容调整）：

## 🎯 一句话理解
[用一句话概括这个概念的直观理解]

## 🧩 核心原理
[讲清楚它是什么、怎么工作，尽量用类比帮助理解]

## 🔧 在 RAG 中的作用 / 做法
[它在完整 RAG 流程中处于哪一步、解决什么问题、常见做法]

## 💡 易错点 / 补充
[仅当教程或你的理解中有值得提醒的点时包含，可省略]

注意：
- 严格以教程内容为准，不要编造教程里没有的细节
- 教程没讲到的点要如实说明，再补充自己的理解
- 语气亲切、结构化，适合自学

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 8000) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度（超出部分按文档粒度截断，保证至少放入第一个文档）

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关教程资料。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【教程资料 {i}】"
            if 'doc_title' in doc.metadata:
                metadata_info += f" {doc.metadata['doc_title']}"
            if 'chapter' in doc.metadata:
                metadata_info += f" | 章节: {doc.metadata['chapter']}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 剩余可用空间
            remaining = max_length - current_length
            if remaining <= 0:
                # 空间已满，跳过后续文档
                break
            
            # 单个文档过长时截断，而不是丢弃整个文档
            if len(doc_text) > remaining:
                doc_text = doc_text[:remaining] + "\n...(内容过长已截断)"
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        divider = "\n" + "="*50 + "\n"
        return divider + divider.join(context_parts)
