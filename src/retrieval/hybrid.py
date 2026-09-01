import logging
from typing import List, Optional

from src.config import settings
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import SearchResult
from src.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid Search combining Dense Vector Search and BM25 Sparse Search via RRF."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = settings.rrf_k,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    def close(self) -> None:
        """Close sub-retrievers and release underlying resources."""
        if hasattr(self, "dense_retriever") and self.dense_retriever is not None:
            self.dense_retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def hybrid_search(self, query: str, top_k: int = settings.retrieval_top_k) -> List[SearchResult]:
        """Perform hybrid retrieval over dense and sparse indices using Reciprocal Rank Fusion."""
        if not query.strip():
            return []

        # Retrieve a broader candidate pool from both retrievers to optimize RRF fusion quality
        candidate_k = max(top_k * 2, 20)

        dense_results: List[SearchResult] = []
        try:
            dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        except Exception as e:
            logger.error(f"Dense retrieval error: {e}", exc_info=True)

        sparse_results: List[SearchResult] = []
        try:
            sparse_results = self.sparse_retriever.search(query, top_k=candidate_k)
        except Exception as e:
            logger.error(f"Sparse retrieval error: {e}", exc_info=True)

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
            rrf_k=self.rrf_k,
        )

        return fused_results
