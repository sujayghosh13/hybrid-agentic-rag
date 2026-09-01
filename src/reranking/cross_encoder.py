import logging
from typing import Any, List, Optional, Union

import numpy as np
from src.config import settings
from sentence_transformers import CrossEncoder

from src.reranking.models import RerankedResult
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder re-ranking engine for scoring and ordering candidate chunks."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
    ):
        self.model_name = model_name or settings.reranker_model_name

        if model is not None:
            self.model = model
        else:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Union[SearchResult, RerankedResult]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Re-rank candidate documents against the user query using Cross-Encoder scores.

        Args:
            query: The user query string.
            candidates: Candidate search results from hybrid retrieval.
            top_k: Number of top re-ranked results to return (defaults to settings.rerank_top_k).

        Returns:
            List of RerankedResult objects sorted descending by rerank_score.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

        final_top_k = top_k if top_k is not None else settings.rerank_top_k

        if not query or not query.strip():
            logger.warning("Empty query provided to CrossEncoderReranker. Returning empty list.")
            return []

        if not candidates:
            return []

        # Construct query-document pairs
        pairs = [(query.strip(), cand.text) for cand in candidates]

        # Compute cross-attention relevance scores
        raw_scores = self.model.predict(pairs)
        if isinstance(raw_scores, (int, float)):
            scores = [float(raw_scores)]
        elif isinstance(raw_scores, np.ndarray):
            scores = raw_scores.tolist()
            if not isinstance(scores, list):
                scores = [float(scores)]
        else:
            scores = [float(s) for s in raw_scores]

        # Build RerankedResult objects preserving all Phase 2 metadata
        reranked_items: List[RerankedResult] = []
        for cand, score_val in zip(candidates, scores):
            res = RerankedResult(
                chunk_id=cand.chunk_id,
                text=cand.text,
                source=cand.source,
                metadata=cand.metadata,
                score=float(score_val),
                dense_rank=cand.dense_rank,
                sparse_rank=cand.sparse_rank,
                rrf_score=cand.rrf_score,
                rerank_score=float(score_val),
                rerank_rank=None,
            )
            reranked_items.append(res)

        # Sort descending by rerank_score (tie-break deterministically by chunk_id)
        reranked_items.sort(key=lambda x: (-x.rerank_score, x.chunk_id))

        # Assign 1-indexed rerank_rank
        for rank, item in enumerate(reranked_items, start=1):
            item.rerank_rank = rank

        return reranked_items[:final_top_k]
