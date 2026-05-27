from enum import Enum
from typing import Any, Dict, List, Optional
import os
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalStrategy(Enum):
    PRECISE = "precise"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


# Strategy-aware defaults
_STRATEGY_DEFAULTS = {
    RetrievalStrategy.PRECISE: {
        "base_k": 3,
        "k_values": [2, 3, 4],
        "max_results": 5,
        "length_thresholds": (5, 30),
    },
    RetrievalStrategy.BALANCED: {
        "base_k": 5,
        "k_values": [3, 5, 8],
        "max_results": 10,
        "length_thresholds": (10, 50),
    },
    RetrievalStrategy.EXPLORATORY: {
        "base_k": 8,
        "k_values": [5, 8, 12],
        "max_results": 15,
        "length_thresholds": (15, 60),
    },
}


class VectorStoreManager:
    def __init__(
        self,
        huggingface_api_key: str,
        default_strategy: RetrievalStrategy = RetrievalStrategy.BALANCED,
    ):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = None
        self.total_documents = 0
        self.default_strategy = default_strategy
        self.last_retrieval_mode = None
        self.last_retrieval_stats = {}

    def create_vector_store(self, documents: List[str], store_name: str):
        try:
            print("Creating vector store...")
            start_time = time.time()

            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.total_documents = len(documents)
            self.save_vector_store(store_name)

            end_time = time.time()
            print(f"Vector store created in {end_time - start_time:.2f}s ({self.total_documents} documents)")

        except Exception as e:
            print(f"Failed to create vector store: {str(e)}")
            raise

    def save_vector_store(self, store_name: str):
        if self.vector_store:
            try:
                os.makedirs("vector_stores", exist_ok=True)
                self.vector_store.save_local(f"vector_stores/{store_name}")
            except Exception as e:
                print(f"Failed to save vector store: {str(e)}")
                raise

    def load_vector_store(self, store_name: str):
        if os.path.exists(f"vector_stores/{store_name}"):
            try:
                self.vector_store = FAISS.load_local(
                    f"vector_stores/{store_name}",
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return True
            except Exception as e:
                print(f"Failed to load vector store: {str(e)}")
                return False
        return False

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        if self.vector_store:
            try:
                return self.vector_store.similarity_search(query, k=k)
            except Exception as e:
                print(f"Similarity search failed: {str(e)}")
                return []
        return []

    def enhanced_similarity_search(
        self,
        query: str,
        k_values: Optional[List[int]] = None,
        max_results: Optional[int] = None,
    ) -> List[str]:
        if not self.vector_store:
            return []

        if k_values is None:
            defaults = _STRATEGY_DEFAULTS[self.default_strategy]
            k_values = list(defaults["k_values"])

        if max_results is None:
            max_results = _STRATEGY_DEFAULTS[self.default_strategy]["max_results"]

        try:
            print(f"Starting enhanced retrieval for query: {query}")
            all_results = []
            seen_content = set()

            for k in k_values:
                try:
                    results = self.vector_store.similarity_search(query, k=k)
                    for result in results:
                        content_key = result.page_content.strip()
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            all_results.append(result)
                            if len(all_results) >= max_results:
                                break
                    if len(all_results) >= max_results:
                        break
                except Exception as e:
                    print(f"Enhanced retrieval failed for k={k}: {str(e)}")
                    continue

            print(f"Enhanced retrieval collected {len(all_results)} documents")
            return all_results

        except Exception as e:
            print(f"Enhanced retrieval failed: {str(e)}")
            return self.similarity_search(query, k=5)

    def rerank_results(self, query: str, documents: List) -> List:
        if not documents:
            return []

        try:
            print(f"Reranking {len(documents)} documents")
            query_embedding = self.embeddings.embed_query(query)
            scored_docs = []

            for index, doc in enumerate(documents):
                try:
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]

                    doc_length = len(doc.page_content)
                    if doc_length < 50:
                        length_factor = 0.7
                    elif doc_length > 2000:
                        length_factor = 0.9
                    else:
                        length_factor = 1.0

                    final_score = similarity * length_factor
                    scored_docs.append((final_score, doc))
                except Exception as e:
                    print(f"Failed to rerank document {index}: {str(e)}")
                    scored_docs.append((0.5, doc))

            scored_docs.sort(key=lambda item: item[0], reverse=True)
            return [doc for _, doc in scored_docs]

        except Exception as e:
            print(f"Reranking failed: {str(e)}")
            return documents

    def adaptive_k_selection(
        self,
        query: str,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> int:
        strategy = strategy or self.default_strategy
        defaults = _STRATEGY_DEFAULTS[strategy]
        base_k = defaults["base_k"]
        short_thresh, long_thresh = defaults["length_thresholds"]

        query_length = len(query)
        if query_length < short_thresh:
            k = base_k - 1
        elif query_length > long_thresh:
            k = base_k + 2
        else:
            k = base_k

        # Question-word detection (Chinese + English)
        question_words = [
            "什么", "怎么", "为什么", "如何", "哪些", "哪个", "何处", "何时",
            "what", "how", "why", "which", "where", "when", "who",
        ]
        if any(word in query.lower() for word in question_words):
            k += 1

        # Multi-part / comparison queries
        if query.count("和") > 2 or query.count("或") > 2:
            k += 1
        if query.lower().count(" and ") > 2 or query.lower().count(" or ") > 2:
            k += 1

        # Document corpus awareness: never request more than available
        if self.total_documents > 0:
            k = min(k, self.total_documents)

        max_k = _STRATEGY_DEFAULTS[strategy]["max_results"]
        return min(k, max_k)

    def get_basic_context(
        self,
        query: str,
        k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> str:
        if k is None:
            k = self.adaptive_k_selection(query, strategy=strategy)
        documents = self.similarity_search(query, k=k)
        self.last_retrieval_mode = "basic"
        self.last_retrieval_stats = {
            "mode": "basic",
            "requested_k": k,
            "returned_docs": len(documents),
        }
        return "\n".join(doc.page_content for doc in documents)

    def get_enhanced_context(
        self,
        query: str,
        max_tokens: int = 1500,
        strategy: Optional[RetrievalStrategy] = None,
        k_values: Optional[List[int]] = None,
        max_results: Optional[int] = None,
    ) -> str:
        try:
            strategy = strategy or self.default_strategy
            defaults = _STRATEGY_DEFAULTS[strategy]

            adaptive_k = self.adaptive_k_selection(query, strategy=strategy)
            if k_values is None:
                # Build tiered k_values centered on adaptive_k
                tier_low = max(2, adaptive_k - 2)
                tier_mid = adaptive_k
                tier_high = min(adaptive_k + 3, defaults["max_results"])
                k_values = [tier_low, tier_mid, tier_high]

            documents = self.enhanced_similarity_search(
                query,
                k_values=k_values,
                max_results=max_results,
            )
            reranked_docs = self.rerank_results(query, documents)

            processed_context = []
            total_length = 0

            for doc in reranked_docs:
                doc_content = doc.page_content.strip()
                doc_length = len(doc_content)
                if total_length + doc_length <= max_tokens:
                    processed_context.append(doc_content)
                    total_length += doc_length
                else:
                    remaining_space = max_tokens - total_length
                    if remaining_space > 100:
                        processed_context.append(doc_content[:remaining_space])
                    break

            context = "\n---\n".join(processed_context)
            self.last_retrieval_mode = "enhanced"
            self.last_retrieval_stats = {
                "mode": "enhanced",
                "strategy": strategy.value,
                "adaptive_k": adaptive_k,
                "k_values": k_values,
                "candidate_docs": len(documents),
                "context_docs": len(processed_context),
                "context_length": len(context),
            }
            print(f"Enhanced context built with stats: {self.last_retrieval_stats}")
            return context

        except Exception as e:
            print(f"Failed to build enhanced context: {str(e)}")
            return self.get_basic_context(query, strategy=strategy)

    def build_context(
        self,
        query: str,
        use_enhanced: bool = True,
        max_tokens: int = 1500,
        strategy: Optional[RetrievalStrategy] = None,
        k_values: Optional[List[int]] = None,
        max_results: Optional[int] = None,
    ) -> str:
        if use_enhanced:
            return self.get_enhanced_context(
                query,
                max_tokens=max_tokens,
                strategy=strategy,
                k_values=k_values,
                max_results=max_results,
            )
        return self.get_basic_context(query, strategy=strategy)
