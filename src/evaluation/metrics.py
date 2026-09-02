import math
import re
from typing import Any, Collection, List, Sequence, Set


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Compute Hit Rate@K. Returns 1.0 if at least one relevant chunk appears in top-K, else 0.0."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    return 1.0 if any(doc_id in rel_set for doc_id in top_k) else 0.0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Compute Recall@K. Returns fraction of all relevant chunks found in top-K."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return hits / float(len(rel_set))


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Compute Precision@K.

    Returns the number of relevant chunks in top-K divided strictly by K.
    Penalizes runs returning fewer than K items when K were requested.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return hits / float(k)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int = 10) -> float:
    """Compute Reciprocal Rank (RR) over the top-K retrieved candidates."""
    if k <= 0 or not relevant_ids:
        return 0.0
    rel_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K) using binary relevance."""
    if k <= 0 or not relevant_ids:
        return 0.0

    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # Compute DCG@K
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in rel_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Compute Ideal DCG@K
    idcg = 0.0
    ideal_hits = min(len(rel_set), k)
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aspect_coverage_score(text: str, expected_aspects: Sequence[str]) -> float:
    """Compute fraction of expected technical aspects covered in generated text."""
    if not expected_aspects:
        return 1.0
    if not text or not text.strip():
        return 0.0

    normalized_text = text.lower()
    matched = 0
    for aspect in expected_aspects:
        # Check aspect words/tokens presence
        aspect_lower = aspect.lower().strip()
        if aspect_lower in normalized_text:
            matched += 1
        else:
            # Fallback to key terms intersection
            words = [w for w in re.split(r"\W+", aspect_lower) if len(w) > 3]
            if words and all(w in normalized_text for w in words):
                matched += 1

    return matched / float(len(expected_aspects))


def source_attribution_coverage(sources: Sequence[Any], corpus_chunk_ids: Set[str]) -> float:
    """Compute the fraction of returned sources that map to valid corpus chunk IDs."""
    if not sources:
        return 1.0

    valid = 0
    for s in sources:
        chunk_id = s.get("chunk_id") if isinstance(s, dict) else getattr(s, "chunk_id", None)
        if chunk_id and chunk_id in corpus_chunk_ids:
            valid += 1
    return valid / float(len(sources))
