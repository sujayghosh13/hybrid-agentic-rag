"""Capstone Structural Compatibility Wrapper for Retrievers.

Re-exports hybrid, dense, and sparse (BM25) retrievers from `src.retrieval`
to satisfy capstone specification Section 22.
"""

from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import SearchResult
from src.retrieval.sparse import BM25Retriever

# Alias BM25Retriever as SparseRetriever for generic compatibility
SparseRetriever = BM25Retriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "SearchResult",
    "SparseRetriever",
    "reciprocal_rank_fusion",
]
