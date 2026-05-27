import json
from typing import Any, Dict

import requests


class QwenClient:
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": {"prompt": prompt},
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        response_data = response.json()

        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API request failed: {response.status_code} - {response.text}",
            )

        return response_data.get("output", {}).get("text", "")

    @staticmethod
    def extract_json_block(text: str) -> Dict[str, Any]:
        start_index = text.find("{")
        end_index = text.rfind("}") + 1
        if start_index == -1 or end_index <= start_index:
            raise ValueError("No JSON object found in model output.")
        return json.loads(text[start_index:end_index])
