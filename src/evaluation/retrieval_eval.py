import logging
from typing import Dict, List, Optional, Sequence

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.evaluation.models import BenchmarkQuery, RetrievalMetrics
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)

STANDARD_K_VALUES = [1, 3, 5, 10]


class RetrievalEvaluator:
    """Evaluates and compares Dense, BM25, Hybrid RRF, and Cross-Encoder retrieval pipelines."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
        )
        self.reranker = reranker or CrossEncoderReranker()

    def evaluate_retriever(
        self,
        queries: Sequence[BenchmarkQuery],
        pipeline_name: str,
        retrieved_ids_map: Dict[str, List[str]],
        k_values: Sequence[int] = STANDARD_K_VALUES,
    ) -> RetrievalMetrics:
        """Compute aggregate retrieval metrics over non-refusal queries."""
        evaluable_queries = [q for q in queries if not q.expected_refusal and q.relevant_chunk_ids]
        if not evaluable_queries:
            return RetrievalMetrics()

        hit_rates: Dict[int, List[float]] = {k: [] for k in k_values}
        recalls: Dict[int, List[float]] = {k: [] for k in k_values}
        precisions: Dict[int, List[float]] = {k: [] for k in k_values}
        ndcgs: Dict[int, List[float]] = {k: [] for k in k_values}
        mrrs: List[float] = []

        max_k = max(k_values)

        for q in evaluable_queries:
            retrieved = retrieved_ids_map.get(q.id, [])
            rel_ids = q.relevant_chunk_ids

            mrrs.append(reciprocal_rank(retrieved, rel_ids, k=max_k))

            for k in k_values:
                hit_rates[k].append(hit_rate_at_k(retrieved, rel_ids, k=k))
                recalls[k].append(recall_at_k(retrieved, rel_ids, k=k))
                precisions[k].append(precision_at_k(retrieved, rel_ids, k=k))
                ndcgs[k].append(ndcg_at_k(retrieved, rel_ids, k=k))

        n_q = float(len(evaluable_queries))
        return RetrievalMetrics(
            hit_rate={k: sum(hit_rates[k]) / n_q for k in k_values},
            recall={k: sum(recalls[k]) / n_q for k in k_values},
            precision={k: sum(precisions[k]) / n_q for k in k_values},
            mrr=sum(mrrs) / n_q if mrrs else 0.0,
            ndcg={k: sum(ndcgs[k]) / n_q for k in k_values},
        )

    def run_all_retrievers(
        self,
        queries: Sequence[BenchmarkQuery],
        top_k: int = 10,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Execute all 4 retrieval pipelines across queries and collect retrieved chunk IDs.

        Returns:
            Dict mapping pipeline_name -> {query_id -> list_of_chunk_ids}
        """
        dense_map: Dict[str, List[str]] = {}
        sparse_map: Dict[str, List[str]] = {}
        hybrid_map: Dict[str, List[str]] = {}
        reranked_map: Dict[str, List[str]] = {}

        candidate_pool_k = max(top_k * 2, 20)

        for q in queries:
            # 1. Dense retrieval
            dense_res = self.dense_retriever.search(q.query, top_k=top_k)
            dense_map[q.id] = [r.chunk_id for r in dense_res]

            # 2. Sparse (BM25) retrieval
            sparse_res = self.sparse_retriever.search(q.query, top_k=top_k)
            sparse_map[q.id] = [r.chunk_id for r in sparse_res]

            # 3. Hybrid RRF retrieval
            hybrid_res = self.hybrid_retriever.hybrid_search(q.query, top_k=candidate_pool_k)
            hybrid_map[q.id] = [r.chunk_id for r in hybrid_res[:top_k]]

            # 4. Cross-Encoder Reranked
            reranked_res = self.reranker.rerank(q.query, candidates=hybrid_res, top_k=top_k)
            reranked_map[q.id] = [r.chunk_id for r in reranked_res]

        return {
            "dense": dense_map,
            "sparse": sparse_map,
            "hybrid_rrf": hybrid_map,
            "hybrid_reranked": reranked_map,
        }
