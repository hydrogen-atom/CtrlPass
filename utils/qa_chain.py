import time
from typing import Any, Optional

import requests

from utils.vector_store import RetrievalStrategy


class QAChain:
    def __init__(
        self,
        qwen_api_key: str,
        vector_store: Any,
        finetuned_model_path: Optional[str] = None,
        use_enhanced_retrieval: bool = True,
        retrieval_strategy: Optional[RetrievalStrategy] = None,
    ):
        self.api_key = qwen_api_key
        self.vector_store = vector_store
        self.use_enhanced_retrieval = use_enhanced_retrieval
        self.retrieval_strategy = retrieval_strategy
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.use_finetuned = finetuned_model_path is not None
        if self.use_finetuned:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading finetuned model from {finetuned_model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                finetuned_model_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                finetuned_model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            print("Finetuned model loaded.")

    def _build_context(self, question: str, strategy: Optional[RetrievalStrategy] = None) -> str:
        strategy = strategy or self.retrieval_strategy
        if hasattr(self.vector_store, "build_context"):
            return self.vector_store.build_context(
                question,
                use_enhanced=self.use_enhanced_retrieval,
                max_tokens=1500,
                strategy=strategy,
            )

        documents = self.vector_store.similarity_search(question, k=5)
        return "\n".join(doc.page_content for doc in documents)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""你是一位经验丰富的学习助手。以下是本次考试的复习资料。请根据这些复习资料回答问题，并给出相关复习建议。
如果参考信息中没有相关内容，请根据现有知识给出合理回答，并明确说明该信息未在复习资料中找到。

参考信息：
{context}

问题：{question}

回答："""

    def get_answer(self, question: str, context_override: Optional[str] = None) -> str:
        """Get an answer for the provided question."""
        try:
            print(f"Processing question: {question}")
            start_time = time.time()

            if context_override is not None:
                context = context_override
            else:
                strategy = getattr(self, "retrieval_strategy", None)
                context = self._build_context(question, strategy=strategy)
            prompt = self._build_prompt(question, context)

            if self.use_finetuned:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_length=2048,
                    temperature=0.2,
                    top_p=0.8,
                    top_k=50,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = answer[len(prompt):].strip()
            else:
                payload = {
                    "model": "qwen-turbo",
                    "input": {"prompt": prompt},
                    "parameters": {
                        "temperature": 0.4,
                        "max_tokens": 800,
                    },
                }

                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response_data = response.json()

                if response.status_code == 200:
                    answer = response_data.get("output", {}).get("text", "无法获取回答")
                else:
                    error_msg = f"API 调用失败: {response.status_code} - {response.text}"
                    print(error_msg)
                    return error_msg

            elapsed = time.time() - start_time
            print(f"Question processed in {elapsed:.2f}s")
            return answer

        except Exception as exc:
            error_msg = f"获取答案时出错: {exc}"
            print(error_msg)
            return error_msg
