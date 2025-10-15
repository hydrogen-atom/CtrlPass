from typing import List, Dict, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.memory import ConversationBufferMemory
from .knowledge_base import KnowledgeBase

class QAChain:
    def __init__(self, qianfan_api_key: str, qianfan_secret_key: str):
        """
        初始化QA链
        Args:
            qianfan_api_key: 通义千问 API Key
            qianfan_secret_key: 通义千问 Secret Key
        """
        self.knowledge_base = KnowledgeBase()
        
        # 初始化LLM
        self.llm = QianfanChatEndpoint(
            model="qianfan-chinese-llama-2-7b",
            temperature=0.7,
            qianfan_ak=qianfan_api_key,
            qianfan_sk=qianfan_secret_key
        )
        
        # 初始化对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 初始化QA链
        self.qa_chain = None
        
        # 问题类型提示模板
        self.question_type_prompts = {
            "factual": """请基于以下文档内容回答问题，同时可以结合你的知识进行补充说明。
            1. 首先使用文档中明确提到的信息作为主要依据
            2. 如果文档信息不足，可以补充你的相关知识
            3. 如果使用了自己的知识，请明确标注"补充说明："
            4. 确保回答准确、完整、有深度""",
            
            "inferential": """请基于以下文档内容进行推理分析，并结合你的知识进行深入解读。
            1. 使用文档内容作为推理的基础
            2. 结合你的知识进行合理的延伸和解释
            3. 区分文档中的信息和你的推理
            4. 提供有深度的分析和见解""",
            
            "summary": """请对以下文档内容进行总结，并加入你的专业见解。
            1. 提取文档中的关键信息
            2. 补充相关的背景知识
            3. 提供专业的分析和建议
            4. 确保总结全面且有深度""",
            
            "comparison": """请比较以下文档内容中的不同观点或方法，并加入你的专业分析。
            1. 基于文档内容进行对比
            2. 补充相关的专业知识和经验
            3. 提供深入的分析和见解
            4. 给出专业的建议和结论""",
            
            "definition": """请解释以下文档内容中的概念或术语，并补充相关知识。
            1. 使用文档中的定义作为基础
            2. 补充相关的专业解释和例子
            3. 提供更广泛的应用场景
            4. 确保解释准确且易于理解""",
            
            "procedural": """请详细说明以下文档内容中描述的过程或步骤，并补充最佳实践。
            1. 基于文档内容描述基本流程
            2. 补充相关的专业经验和技巧
            3. 提供实用的建议和注意事项
            4. 确保说明清晰且可操作""",
            
            "opinion": """请分析以下文档内容中的观点和论证，并提供专业的见解。
            1. 基于文档内容分析主要观点
            2. 补充相关的专业知识和经验
            3. 提供深入的分析和评价
            4. 给出专业的建议和结论"""
        }
    
    def add_document(self, file_path: str) -> Dict:
        """
        添加文档到知识库
        Args:
            file_path: 文档路径
        Returns:
            Dict: 添加结果信息
        """
        return self.knowledge_base.add_document(file_path)
    
    def _create_qa_chain(self):
        """创建QA链"""
        if self.knowledge_base.vectorstore is None:
            return None
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.knowledge_base.vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.7  # 设置相似度阈值
                }
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": self._get_qa_prompt()
            }
        )
    
    def _get_qa_prompt(self, question_type: str = "factual") -> str:
        """获取问答提示模板"""
        base_prompt = self.question_type_prompts.get(question_type, self.question_type_prompts["factual"])
        return f"""
        {base_prompt}
        
        文档内容：
        {{context}}
        
        问题：{{question}}
        
        请基于文档内容回答问题，同时可以结合你的知识进行补充说明。
        回答时请：
        1. 首先引用相关的文档内容作为支持
        2. 然后可以补充你的专业知识和见解
        3. 如果使用了自己的知识，请明确标注"补充说明："
        4. 确保回答准确、完整、有深度
        """
    
    def answer_question(self, question: str, question_type: str = "factual") -> Dict:
        """
        回答问题
        Args:
            question: 问题
            question_type: 问题类型
        Returns:
            Dict: 回答结果
        """
        # 如果知识库为空，重新添加文档
        if self.knowledge_base.vectorstore is None:
            return {
                "status": "error",
                "message": "知识库为空，请先添加文档"
            }
        
        # 创建或获取QA链
        if self.qa_chain is None:
            self.qa_chain = self._create_qa_chain()
            if self.qa_chain is None:
                return {
                    "status": "error",
                    "message": "无法创建QA链"
                }
        
        try:
            # 更新提示模板
            self.qa_chain.combine_docs_chain.llm_chain.prompt = self._get_qa_prompt(question_type)
            
            # 执行问答
            result = self.qa_chain({"question": question})
            
            # 格式化结果
            return {
                "status": "success",
                "answer": result["answer"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": doc.metadata.get("score", 0)
                    }
                    for doc in result["source_documents"]
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"回答问题时出错: {str(e)}"
            }
    
    def clear_memory(self):
        """清除对话记忆"""
        self.memory.clear()
        self.qa_chain = None
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        Returns:
            Dict: 统计信息
        """
        return self.knowledge_base.get_stats() 