from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SearchResult:
    """Represents a candidate search result retrieved from vector, BM25, or hybrid search."""

    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize search result to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf_score": self.rrf_score,
        }
