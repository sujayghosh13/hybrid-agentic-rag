from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RerankedResult:
    """Represents a candidate result re-scored and ranked by a Cross-Encoder model."""

    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: float = 0.0
    rerank_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reranked result to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
        }
