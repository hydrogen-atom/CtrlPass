import os
import re
from dataclasses import dataclass
from typing import List

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

SENTENCE_SEPARATORS = ["。", "；", "！", "？", ".", "!", "?", "\n"]
TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", ".", "!", "?", " ", ""]
SENTENCE_BOUNDARY_PATTERN = r"(?<=[。；！？.!?\n])"


@dataclass
class Chunk:
    text: str
    start_idx: int
    end_idx: int
    score: float = 0.0


class ModelBasedTextSplitter:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "Model-based splitting requires sentence-transformers and a working torch installation."
            ) from exc

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the model-based splitter. Please check the local torch/runtime installation."
            ) from exc

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences, show_progress_bar=False)

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _split_into_sentences(self, text: str) -> List[str]:
        return [part.strip() for part in re.split(SENTENCE_BOUNDARY_PATTERN, text) if part.strip()]

    def split_text(self, text: str) -> List[Chunk]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [Chunk(text=text, start_idx=0, end_idx=len(text))]

        sentence_embeddings = self._get_sentence_embeddings(sentences)
        chunks = []
        current_chunk = []
        current_start = 0
        current_length = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            if not current_chunk:
                current_chunk.append(sentence)
                current_length = len(sentence)
                continue

            last_embedding = sentence_embeddings[i - 1]
            similarity = self._calculate_similarity(embedding, last_embedding)

            if similarity > self.similarity_threshold and current_length + len(sentence) <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                chunk_text = "".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start_idx=current_start,
                            end_idx=current_start + len(chunk_text),
                            score=similarity,
                        )
                    )

                overlap_sentences = []
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    if overlap_length + len(prev_sentence) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence)
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_start = current_start + len(chunk_text) - overlap_length
                current_length = overlap_length + len(sentence)

        if current_chunk:
            chunk_text = "".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_idx=current_start,
                        end_idx=current_start + len(chunk_text),
                    )
                )

        return chunks


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_model_splitter: bool = False,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_model_splitter = use_model_splitter

        if use_model_splitter:
            self.model_splitter = ModelBasedTextSplitter(
                model_name=model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=TEXT_SPLITTER_SEPARATORS,
                keep_separator=True,
            )

    def load_document(self, file_path: str) -> List[str]:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return loader.load()

    def split_documents(self, documents: List[str]) -> List[str]:
        """Split documents into smaller chunks."""
        if self.use_model_splitter:
            all_chunks = []
            for doc in documents:
                chunks = self.model_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    new_doc = type(doc)(
                        page_content=chunk.text,
                        metadata=doc.metadata.copy(),
                    )
                    all_chunks.append(new_doc)
            return all_chunks

        return self.text_splitter.split_documents(documents)
