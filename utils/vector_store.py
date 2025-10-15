from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import List
import os
import time

class VectorStoreManager:
    def __init__(self, huggingface_api_key: str):
        # 设置Hugging Face token环境变量
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
        
        # 使用轻量级的开源embeddings模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = None

    def create_vector_store(self, documents: List[str], store_name: str):
        """创建向量存储"""
        try:
            # 添加进度提示
            print("开始创建向量存储...")
            start_time = time.time()
            
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.save_vector_store(store_name)
            
            end_time = time.time()
            print(f"向量存储创建完成，耗时: {end_time - start_time:.2f}秒")
            
        except Exception as e:
            print(f"创建向量存储时出错: {str(e)}")
            raise

    def save_vector_store(self, store_name: str):
        """保存向量存储到文件"""
        if self.vector_store:
            try:
                os.makedirs("vector_stores", exist_ok=True)
                self.vector_store.save_local(f"vector_stores/{store_name}")
            except Exception as e:
                print(f"保存向量存储时出错: {str(e)}")
                raise

    def load_vector_store(self, store_name: str):
        """从文件加载向量存储"""
        if os.path.exists(f"vector_stores/{store_name}"):
            try:
                self.vector_store = FAISS.load_local(f"vector_stores/{store_name}", self.embeddings)
                return True
            except Exception as e:
                print(f"加载向量存储时出错: {str(e)}")
                return False
        return False

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """执行相似度搜索"""
        if self.vector_store:
            try:
                return self.vector_store.similarity_search(query, k=k)
            except Exception as e:
                print(f"执行相似度搜索时出错: {str(e)}")
                return []
        return [] 