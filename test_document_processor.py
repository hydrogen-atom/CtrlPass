import unittest
from unittest.mock import patch

import numpy as np
from langchain_core.documents import Document

from utils.document_processor import DocumentProcessor, ModelBasedTextSplitter


class DocumentProcessorTests(unittest.TestCase):
    def test_model_splitter_splits_chinese_sentences(self):
        splitter = ModelBasedTextSplitter.__new__(ModelBasedTextSplitter)

        text = "机器学习帮助计算机从数据中学习。监督学习使用标签？深度学习可以处理复杂任务！"
        sentences = splitter._split_into_sentences(text)

        self.assertEqual(
            sentences,
            [
                "机器学习帮助计算机从数据中学习。",
                "监督学习使用标签？",
                "深度学习可以处理复杂任务！",
            ],
        )

    def test_model_splitter_returns_no_chunks_for_short_text_below_min_size(self):
        splitter = ModelBasedTextSplitter.__new__(ModelBasedTextSplitter)
        splitter.chunk_size = 500
        splitter.chunk_overlap = 100
        splitter.similarity_threshold = 0.7
        splitter.min_chunk_size = 100

        with patch.object(
            splitter,
            "_get_sentence_embeddings",
            return_value=np.array([[1.0, 0.0], [1.0, 0.0]]),
        ):
            chunks = splitter.split_text("机器学习帮助理解数据。监督学习使用标签。")

        self.assertEqual(chunks, [])

    def test_model_splitter_generates_chunks_for_chinese_text(self):
        splitter = ModelBasedTextSplitter.__new__(ModelBasedTextSplitter)
        splitter.chunk_size = 40
        splitter.chunk_overlap = 0
        splitter.similarity_threshold = 0.7
        splitter.min_chunk_size = 10

        with patch.object(
            splitter,
            "_get_sentence_embeddings",
            return_value=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        ):
            chunks = splitter.split_text(
                "机器学习是一种方法。监督学习依赖标签。深度学习处理复杂任务。"
            )

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk.text for chunk in chunks))

    def test_default_splitter_uses_chinese_punctuation_boundaries(self):
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=0, use_model_splitter=False)
        documents = [
            Document(
                page_content="机器学习是一种方法。监督学习依赖标签。深度学习处理复杂任务。",
                metadata={"source": "sample"},
            )
        ]

        chunks = processor.split_documents(documents)
        chunk_texts = [chunk.page_content for chunk in chunks]

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunk_texts[0], "机器学习是一种方法")
        self.assertEqual(chunk_texts[1], "。监督学习依赖标签")
        self.assertEqual("".join(chunk_texts), "机器学习是一种方法。监督学习依赖标签。深度学习处理复杂任务。")


if __name__ == "__main__":
    unittest.main()
