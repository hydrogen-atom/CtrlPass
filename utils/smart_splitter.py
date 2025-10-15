from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

class SmartTextSplitter:
    def __init__(self):
        """
        初始化智能分块器
        """
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # 基础分块器
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # 问题类型对应的分块策略
        self.question_strategies = {
            "factual": {
                "chunk_size": 300,  # 较小块，便于精确匹配
                "chunk_overlap": 50,
                "split_by": "sentence"
            },
            "inferential": {
                "chunk_size": 800,  # 较大块，保留更多上下文
                "chunk_overlap": 200,
                "split_by": "paragraph"
            },
            "summary": {
                "chunk_size": 1000,  # 最大块，保留完整段落
                "chunk_overlap": 100,
                "split_by": "paragraph"
            },
            "comparison": {
                "chunk_size": 600,  # 中等块，便于比较
                "chunk_overlap": 150,
                "split_by": "paragraph"
            },
            "definition": {
                "chunk_size": 400,  # 较小块，便于定义
                "chunk_overlap": 50,
                "split_by": "sentence"
            },
            "procedural": {
                "chunk_size": 500,  # 中等块，保留步骤
                "chunk_overlap": 100,
                "split_by": "sentence"
            },
            "opinion": {
                "chunk_size": 700,  # 较大块，保留观点
                "chunk_overlap": 150,
                "split_by": "paragraph"
            }
        }
        
        # 文本特征分析器
        self.text_analyzers = {
            "sentence_length": self._analyze_sentence_length,
            "paragraph_length": self._analyze_paragraph_length,
            "technical_terms": self._analyze_technical_terms,
            "code_blocks": self._analyze_code_blocks
        }

    def _analyze_sentence_length(self, text: str) -> Dict:
        """分析句子长度特征"""
        sentences = sent_tokenize(text)
        lengths = [len(s) for s in sentences]
        return {
            "avg_length": np.mean(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "std_length": np.std(lengths)
        }

    def _analyze_paragraph_length(self, text: str) -> Dict:
        """分析段落长度特征"""
        paragraphs = text.split('\n\n')
        lengths = [len(p) for p in paragraphs if p.strip()]
        return {
            "avg_length": np.mean(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "std_length": np.std(lengths)
        }

    def _analyze_technical_terms(self, text: str) -> Dict:
        """分析技术术语密度"""
        # 简单的技术术语检测（可以根据需要扩展）
        technical_patterns = [
            r'\b(?:function|class|method|algorithm|protocol|interface)\b',
            r'\b(?:API|SDK|REST|HTTP|TCP|IP)\b',
            r'\b(?:database|server|client|network|security)\b'
        ]
        
        matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                     for pattern in technical_patterns)
        total_words = len(text.split())
        
        return {
            "technical_density": matches / total_words if total_words > 0 else 0,
            "total_terms": matches
        }

    def _analyze_code_blocks(self, text: str) -> Dict:
        """分析代码块特征"""
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        return {
            "has_code": len(code_blocks) > 0,
            "code_blocks_count": len(code_blocks),
            "code_ratio": sum(len(block) for block in code_blocks) / len(text) if text else 0
        }

    def analyze_text(self, text: str) -> Dict:
        """分析文本特征"""
        features = {}
        for analyzer_name, analyzer_func in self.text_analyzers.items():
            features[analyzer_name] = analyzer_func(text)
        return features

    def adjust_strategy(self, strategy: Dict, text_features: Dict) -> Dict:
        """根据文本特征调整分块策略"""
        adjusted_strategy = strategy.copy()
        
        # 根据句子长度调整
        if text_features["sentence_length"]["avg_length"] > 100:
            adjusted_strategy["chunk_size"] = min(
                adjusted_strategy["chunk_size"] * 1.5,
                2000
            )
        
        # 根据技术术语密度调整
        if text_features["technical_terms"]["technical_density"] > 0.1:
            adjusted_strategy["chunk_size"] = min(
                adjusted_strategy["chunk_size"] * 1.2,
                1500
            )
            adjusted_strategy["chunk_overlap"] = min(
                adjusted_strategy["chunk_overlap"] * 1.5,
                300
            )
        
        # 根据代码块调整
        if text_features["code_blocks"]["has_code"]:
            adjusted_strategy["split_by"] = "code"
            adjusted_strategy["chunk_size"] = max(
                adjusted_strategy["chunk_size"] * 1.3,
                1000
            )
        
        return adjusted_strategy

    def split_text(self, text: str, question_type: str = "factual") -> List[str]:
        """
        智能分块文本
        Args:
            text: 要分块的文本
            question_type: 问题类型
        Returns:
            List[str]: 分块后的文本列表
        """
        # 获取基础策略
        strategy = self.question_strategies.get(
            question_type,
            self.question_strategies["factual"]
        )
        
        # 分析文本特征
        text_features = self.analyze_text(text)
        
        # 调整策略
        adjusted_strategy = self.adjust_strategy(strategy, text_features)
        
        # 创建分块器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=adjusted_strategy["chunk_size"],
            chunk_overlap=adjusted_strategy["chunk_overlap"],
            length_function=len,
        )
        
        # 根据策略选择分块方式
        if adjusted_strategy["split_by"] == "sentence":
            # 按句子分块
            chunks = splitter.split_text(text)
        elif adjusted_strategy["split_by"] == "paragraph":
            # 按段落分块
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                if current_length + len(para) > adjusted_strategy["chunk_size"]:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_length = len(para)
                else:
                    current_chunk.append(para)
                    current_length += len(para)
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        elif adjusted_strategy["split_by"] == "code":
            # 特殊处理代码块
            chunks = []
            current_chunk = []
            current_length = 0
            
            # 分割文本，保持代码块完整
            parts = re.split(r'(```[\s\S]*?```)', text)
            
            for part in parts:
                if part.startswith('```'):
                    # 代码块
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    chunks.append(part)
                    current_chunk = []
                    current_length = 0
                else:
                    # 普通文本
                    if current_length + len(part) > adjusted_strategy["chunk_size"]:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [part]
                        current_length = len(part)
                    else:
                        current_chunk.append(part)
                        current_length += len(part)
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    def get_split_info(self, text: str, question_type: str = "factual") -> Dict:
        """
        获取分块信息
        Args:
            text: 要分块的文本
            question_type: 问题类型
        Returns:
            Dict: 分块信息
        """
        # 分析文本特征
        text_features = self.analyze_text(text)
        
        # 获取基础策略
        strategy = self.question_strategies.get(
            question_type,
            self.question_strategies["factual"]
        )
        
        # 调整策略
        adjusted_strategy = self.adjust_strategy(strategy, text_features)
        
        # 执行分块
        chunks = self.split_text(text, question_type)
        
        return {
            "original_strategy": strategy,
            "adjusted_strategy": adjusted_strategy,
            "text_features": text_features,
            "chunks_count": len(chunks),
            "avg_chunk_size": np.mean([len(chunk) for chunk in chunks]) if chunks else 0,
            "chunks": chunks
        } 