"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量：先当前目录，再加载仓库根目录下的 .env（存放 DEEPSEEK_API_KEY）
load_dotenv()
load_dotenv(Path(__file__).parent / '.env')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGLearningAssistant:
    """RAG 学习问答助手主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载教程文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载教程文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块（传入嵌入模型，用于重排序精排）
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks, self.index_module.embeddings)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   章节分布: {list(stats['chapters'].keys())}")

        print("✅ 知识库构建完成！")
    
    def ask_question(self, question: str, stream: bool = False, return_trace: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出（True 时返回生成器，忽略 return_trace）
            return_trace: 是否返回结构化中间过程（供界面可视化，仅在 stream=False 时生效）

        Returns:
            - 默认返回生成的回答字符串
            - return_trace=True 时返回 (回答, trace字典)
            - stream=True 时返回回答生成器
        """
        import time
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        t0 = time.time()
        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由（本地规则版，省一次 LLM 调用）
        route_type = self._route_query(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（本地判断：明确问题跳过改写，省一次 LLM 调用）
        did_rewrite = False
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        elif self._needs_rewrite(question):
            # 仅模糊/宽泛问题才调用 LLM 改写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)
            did_rewrite = True
        else:
            # 明确问题直接原查询检索，跳过改写
            rewritten_query = question
            print(f"📝 问题明确，跳过改写: {question}")

        # 3. 检索相关子块（宽进：先召回较多候选，自动应用元数据过滤）
        print("🔍 检索相关文档...")
        recall_k = self.config.top_k * 4  # 宽进召回，供后续重排序精排
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=recall_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=recall_k)

        # 3.5 重排序精排：对候选集二次打分，取最相关的 top_k
        if len(relevant_chunks) > self.config.top_k:
            print("🎯 重排序精排...")
            relevant_chunks = self.retrieval_module.rerank_documents(rewritten_query, relevant_chunks, top_n=self.config.top_k)

        # 构建检索块结构化信息（供界面展示 + 日志）
        chunk_info = []
        for chunk in relevant_chunks:
            doc_title = chunk.metadata.get('doc_title', '未知主题')
            chapter = chunk.metadata.get('chapter', '')
            content_preview = chunk.page_content[:100].strip()
            if content_preview.startswith('#'):
                title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                section_title = content_preview[:title_end].replace('#', '').strip()
            else:
                section_title = '内容片段'
            chunk_info.append({
                'doc_title': doc_title,
                'chapter': chapter,
                'section': section_title,
                'preview': chunk.page_content[:200].strip(),
            })

        if chunk_info:
            print(f"找到 {len(relevant_chunks)} 个相关文档块: " + ", ".join([f"{c['doc_title']}({c['section']})" for c in chunk_info]))
        else:
            print(f"找到 0 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            answer = "抱歉，没有找到相关的教程内容。请尝试换一种问法或更具体的关键词。"
            trace = {
                'route_type': route_type,
                'rewritten_query': rewritten_query,
                'did_rewrite': did_rewrite,
                'filters': filters,
                'chunks': chunk_info,
                'doc_names': [],
                'elapsed': round(time.time() - t0, 2),
            }
            return (answer, trace) if return_trace else answer

        # 5. 获取父文档
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        doc_names = []
        for doc in relevant_docs:
            doc_title = doc.metadata.get('doc_title', '未知主题')
            if doc_title not in doc_names:
                doc_names.append(doc_title)

        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")

        # 组装结构化 trace（含生成前耗用的检索时间）
        trace = {
            'route_type': route_type,
            'rewritten_query': rewritten_query,
            'did_rewrite': did_rewrite,
            'filters': filters,
            'chunks': chunk_info,
            'doc_names': doc_names,
            'elapsed': round(time.time() - t0, 2),
        }

        # 6. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回相关主题列表（纯规则，无 LLM）
            print("📋 生成主题列表...")
            answer = self.generation_module.generate_list_answer(question, relevant_docs)
            return (answer, trace) if return_trace else answer
        else:
            # 详细/一般查询：调用 LLM 生成详细回答
            print("✍️ 生成详细回答...")
            if route_type == "detail":
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                answer = self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                answer = self.generation_module.generate_basic_answer(question, relevant_docs)
            return (answer, trace) if return_trace else answer
    
    def _route_query(self, query: str) -> str:
        """
        查询路由（本地规则版）—— 替代 LLM 路由，省一次网络调用

        根据关键词快速判断问题类型：
        - list：盘点/列举类（"有哪些""几种""哪些"）
        - detail：深挖某个概念/做法（"怎么做""为什么""如何""原理""区别"）
        - general：其他（含"是什么"这类概念性问题）

        权衡说明：规则路由比 LLM 路由少一次调用（省 1–3 秒），
        且结果稳定可复现；学习问答的路由信号关键词明确，规则准确率足够高。
        原 LLM 路由 query_router 仍保留在 generation 模块，可作备选。
        """
        q = query.strip()

        # 1. 盘点/列举类 → list
        list_keywords = ['有哪些', '有几种', '几种', '哪些', '列举', '盘点',
                         '多少种', '分类', '有哪些类型', '什么方式', '什么方法', '什么工具']
        if any(kw in q for kw in list_keywords):
            return 'list'

        # 2. 深挖概念/做法类 → detail
        detail_keywords = ['怎么做', '如何', '怎么', '为什么', '原理', '步骤',
                           '流程', '实现', '区别', '对比', '详解', '怎么算']
        if any(kw in q for kw in detail_keywords):
            return 'detail'

        # 3. 默认 → general
        return 'general'

    def _needs_rewrite(self, query: str) -> bool:
        """
        本地规则判断查询是否需要改写（明显模糊才改写，明确则跳过）

        目的：避免为"明确问题"白白多花一次 LLM 改写调用。
        判断原则偏保守——拿不准就改写，保效果；拿得准才跳过，省时间。
        """
        q = query.strip()

        # 过短 → 信息不足，需要改写
        if len(q) < 4:
            return True

        # 含指代词 → 缺上下文，需要改写
        if any(w in q for w in ['这个', '那个', '它', '这', '那', '这块']):
            return True

        # 明显宽泛、无主语 → 需要改写
        if q in ['RAG', '检索', '向量', '生成', '向量数据库', '嵌入', '分块']:
            return True

        # 其余视为明确 → 跳过改写，直接原查询检索
        return False

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件（章节）

        当问题命中某章节的关键词时，优先只在该章节内检索，
        缩小检索空间、减少无关文档的噪声干扰。

        采用"最长关键词优先"：多个关键词同时命中时，取更具体（更长）
        的那个，避免"向量检索"里的"向量"误命中到向量章节。
        """
        # (章节, 关键词列表)，关键词从具体到宽泛排列
        chapter_keywords = [
            ('第4章·检索优化', ['混合检索', '倒数排名', '查询重写', '查询改写', 'text2sql', 'rerank', '重排', '查询构建', '检索']),
            ('第3章·向量嵌入与向量库', ['多模态嵌入', '向量数据库', '索引优化', 'embedding', 'milvus', '嵌入', '向量']),
            ('第2章·数据加载与分块', ['数据加载', '文档加载', 'chunking', 'chunk', '分块', '切分']),
            ('第1章·RAG基础', ['什么是rag', 'rag简介', 'rag是什么', '环境准备', '入门']),
            ('第5章·生成优化', ['结构化输出', 'function calling', '格式化', '生成']),
            ('第6章·系统评估', ['忠实度', '评估', '评测', '指标', 'eval']),
            ('第7章·知识图谱与Agent', ['知识图谱', 'agentic', 'agent', 'kg']),
            ('第8章·项目实战', ['项目', '实战', '架构', '部署', '环境']),
            ('第9章·GraphRAG', ['graphrag', '图谱rag', '图建模', 'routing', '路由']),
        ]

        q = query.lower()

        # 收集所有命中的关键词，取最长者（更具体优先）
        best_keyword_len = 0
        best_chapter = None
        for chapter, keywords in chapter_keywords:
            for kw in keywords:
                if kw in q and len(kw) > best_keyword_len:
                    best_keyword_len = len(kw)
                    best_chapter = chapter

        if best_chapter:
            return {'chapter': best_chapter}
        return {}
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("📚  RAG 学习问答助手 - 交互式问答  📚")
        print("=" * 60)
        print("💡 基于 all-in-rag 教程文档构建，帮你深入理解 RAG 的每一步")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用 RAG 学习问答助手！")


def main():
    """主函数"""
    try:
        # 创建RAG学习助手
        rag_system = RAGLearningAssistant()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
