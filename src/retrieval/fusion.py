from typing import Dict, List, Optional

from src.retrieval.models import SearchResult


def reciprocal_rank_fusion(
    dense_results: List[SearchResult],
    sparse_results: List[SearchResult],
    top_k: int = 10,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """Combine dense and sparse search results using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum(1 / (rrf_k + r_m(d))) across all retrievers m.
    """
    scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}
    payload_store: Dict[str, SearchResult] = {}

    # Process dense results
    for rank, res in enumerate(dense_results, start=1):
        cid = res.chunk_id
        dense_ranks[cid] = rank
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        if cid not in payload_store:
            payload_store[cid] = res

    # Process sparse results
    for rank, res in enumerate(sparse_results, start=1):
        cid = res.chunk_id
        sparse_ranks[cid] = rank
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        if cid not in payload_store:
            payload_store[cid] = res

    # Construct unified SearchResult objects
    fused_results: List[SearchResult] = []
    for cid, rrf_score in scores.items():
        base = payload_store[cid]
        fused_res = SearchResult(
            chunk_id=base.chunk_id,
            text=base.text,
            source=base.source,
            metadata=base.metadata,
            score=rrf_score,
            dense_rank=dense_ranks.get(cid),
            sparse_rank=sparse_ranks.get(cid),
            rrf_score=rrf_score,
        )
        fused_results.append(fused_res)

    # Sort descending by rrf_score (tie-break deterministically by chunk_id)
    fused_results.sort(key=lambda x: (-x.rrf_score, x.chunk_id))

    return fused_results[:top_k]
