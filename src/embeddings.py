"""Capstone Structural Compatibility Wrapper for Vector Embeddings.

Re-exports dense vector retrieval and embedding functionality from
`src.retrieval.dense` to satisfy capstone specification Section 22.
"""

from src.retrieval.dense import DenseRetriever
from src.retrieval.models import SearchResult

__all__ = [
    "DenseRetriever",
    "SearchResult",
]
