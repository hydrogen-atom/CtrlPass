from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
import os

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],  # 添加中文标点作为分隔符
            keep_separator=True  # 保留分隔符
        )

    def load_document(self, file_path: str) -> List[str]:
        """加载文档并返回文本内容"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        documents = loader.load()
        return documents

    def split_documents(self, documents: List[str]) -> List[str]:
        """将文档分割成小块"""
        return self.text_splitter.split_documents(documents) 