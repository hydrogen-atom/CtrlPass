from typing import List, Dict, Optional
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from .smart_splitter import SmartTextSplitter

class KnowledgeBase:
    def __init__(self, persist_directory: str = "vectorstore"):
        """
        初始化知识库
        Args:
            persist_directory: 向量存储持久化目录
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.smart_splitter = SmartTextSplitter()
        
        # 加载已存在的向量存储
        if os.path.exists(persist_directory):
            self.vectorstore = FAISS.load_local(
                persist_directory,
                self.embeddings
            )

    def add_document(self, file_path: str, question_type: str = "factual") -> Dict:
        """
        添加文档到知识库
        Args:
            file_path: 文档路径
            question_type: 问题类型
        Returns:
            Dict: 添加结果信息
        """
        # 根据文件类型选择加载器
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        # 加载文档
        documents = loader.load()
        
        # 使用智能分块器处理文档
        all_chunks = []
        split_info = []
        
        for doc in documents:
            # 获取分块信息
            info = self.smart_splitter.get_split_info(
                doc.page_content,
                question_type
            )
            split_info.append(info)
            
            # 添加分块到列表
            all_chunks.extend(info["chunks"])
        
        # 创建新的向量存储或更新现有的
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                all_chunks,
                self.embeddings
            )
        else:
            self.vectorstore.add_texts(all_chunks)
        
        # 保存向量存储
        self.vectorstore.save_local(self.persist_directory)
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "split_info": split_info
        }
    
    def search(self, query: str, k: int = 4) -> List[Dict]:
        """
        搜索知识库
        Args:
            query: 搜索查询
            k: 返回结果数量
        Returns:
            List[Dict]: 搜索结果
        """
        if self.vectorstore is None:
            return []
        
        # 执行相似度搜索
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=k
        )
        
        # 格式化结果
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """
        获取知识库统计信息
        Returns:
            Dict: 统计信息
        """
        if self.vectorstore is None:
            return {
                "status": "empty",
                "documents_count": 0,
                "vectors_count": 0
            }
        
        return {
            "status": "active",
            "documents_count": len(self.vectorstore.index_to_docstore_id),
            "vectors_count": self.vectorstore.index.ntotal
        }
    
    def clear(self) -> None:
        """清空知识库"""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        self.vectorstore = None

    def _load_knowledge_base_info(self):
        """加载知识库信息"""
        info_path = os.path.join(self.persist_directory, "knowledge_base_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                self.knowledge_base_info = json.load(f)

    def _save_knowledge_base_info(self):
        """保存知识库信息"""
        os.makedirs(self.persist_directory, exist_ok=True)
        info_path = os.path.join(self.persist_directory, "knowledge_base_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base_info, f, ensure_ascii=False, indent=2)

    def create_knowledge_base(self, name: str, description: str) -> bool:
        """
        创建新的知识库
        Args:
            name: 知识库名称
            description: 知识库描述
        Returns:
            bool: 是否创建成功
        """
        try:
            # 检查知识库是否已存在
            if name in self.knowledge_base_info:
                return False
            
            # 创建知识库目录
            kb_path = os.path.join(self.persist_directory, name)
            os.makedirs(kb_path, exist_ok=True)
            
            # 创建空的向量存储
            vector_store = FAISS.from_texts(
                ["这是一个空的知识库"],
                self.embeddings
            )
            
            # 保存向量存储
            vector_store.save_local(kb_path)
            
            # 更新知识库信息
            self.knowledge_base_info[name] = {
                "description": description,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "documents_count": 0,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存知识库信息
            self._save_knowledge_base_info()
            
            return True
            
        except Exception as e:
            print(f"创建知识库时出错: {str(e)}")
            return False

    def get_knowledge_base(self, kb_name: str) -> Optional[FAISS]:
        """
        获取知识库
        Args:
            kb_name: 知识库名称
        Returns:
            Optional[FAISS]: 知识库向量存储
        """
        try:
            # 检查知识库是否存在
            if kb_name not in self.knowledge_base_info:
                return None
            
            # 获取知识库路径
            kb_path = os.path.join(self.persist_directory, kb_name)
            
            # 加载向量存储
            if os.path.exists(kb_path):
                return FAISS.load_local(kb_path, self.embeddings)
            return None
            
        except Exception as e:
            print(f"获取知识库时出错: {str(e)}")
            return None

    def list_knowledge_bases(self) -> List[Dict]:
        """
        列出所有知识库
        Returns:
            List[Dict]: 知识库信息列表
        """
        return [
            {
                "name": name,
                **info
            }
            for name, info in self.knowledge_base_info.items()
        ]

    def delete_knowledge_base(self, kb_name: str) -> bool:
        """
        删除知识库
        Args:
            kb_name: 知识库名称
        Returns:
            bool: 是否删除成功
        """
        try:
            # 检查知识库是否存在
            if kb_name not in self.knowledge_base_info:
                return False
            
            # 删除知识库目录
            kb_path = os.path.join(self.persist_directory, kb_name)
            if os.path.exists(kb_path):
                import shutil
                shutil.rmtree(kb_path)
            
            # 更新知识库信息
            del self.knowledge_base_info[kb_name]
            self._save_knowledge_base_info()
            
            return True
            
        except Exception as e:
            print(f"删除知识库时出错: {str(e)}")
            return False 