import json
import os
from typing import List, Dict
from datetime import datetime

class TrainingDataCollector:
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.qa_pairs: List[Dict[str, str]] = []
        
    def add_qa_pair(self, context: str, question: str, answer: str):
        """添加一个问答对到训练数据中"""
        qa_pair = {
            "context": context,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        self.qa_pairs.append(qa_pair)
        
    def save_data(self, filename: str = None):
        """保存训练数据到文件"""
        if filename is None:
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"训练数据已保存到: {file_path}")
        
    def load_data(self, filename: str) -> List[Dict[str, str]]:
        """从文件加载训练数据"""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
        return self.qa_pairs
    
    def get_statistics(self) -> Dict:
        """获取训练数据统计信息"""
        return {
            "total_pairs": len(self.qa_pairs),
            "avg_context_length": sum(len(qa["context"]) for qa in self.qa_pairs) / len(self.qa_pairs) if self.qa_pairs else 0,
            "avg_question_length": sum(len(qa["question"]) for qa in self.qa_pairs) / len(self.qa_pairs) if self.qa_pairs else 0,
            "avg_answer_length": sum(len(qa["answer"]) for qa in self.qa_pairs) / len(self.qa_pairs) if self.qa_pairs else 0
        } 