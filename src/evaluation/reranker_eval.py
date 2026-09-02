import logging
from typing import Dict, List, Sequence

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from src.evaluation.models import BenchmarkQuery, RerankerMetrics

logger = logging.getLogger(__name__)


class RerankerEvaluator:
    """Isolates and evaluates the ranking quality improvements introduced by Cross-Encoder Reranking."""

    def evaluate_reranker(
        self,
        queries: Sequence[BenchmarkQuery],
        pre_rerank_map: Dict[str, List[str]],
        post_rerank_map: Dict[str, List[str]],
    ) -> RerankerMetrics:
        evaluable_queries = [q for q in queries if not q.expected_refusal and q.relevant_chunk_ids]
        if not evaluable_queries:
            return RerankerMetrics()

        pre_mrrs: List[float] = []
        post_mrrs: List[float] = []
        pre_hit_1s: List[float] = []
        post_hit_1s: List[float] = []
        pre_ndcg_5s: List[float] = []
        post_ndcg_5s: List[float] = []

        promotions = 0
        rank_shifts: List[float] = []

        for q in evaluable_queries:
            pre_ids = pre_rerank_map.get(q.id, [])
            post_ids = post_rerank_map.get(q.id, [])
            rel_ids = set(q.relevant_chunk_ids)

            # MRR (top-10)
            pre_mrrs.append(reciprocal_rank(pre_ids, rel_ids, k=10))
            post_mrrs.append(reciprocal_rank(post_ids, rel_ids, k=10))

            # Hit@1
            pre_hit_1s.append(hit_rate_at_k(pre_ids, rel_ids, k=1))
            post_hit_1s.append(hit_rate_at_k(post_ids, rel_ids, k=1))

            # nDCG@5
            pre_ndcg_5s.append(ndcg_at_k(pre_ids, rel_ids, k=5))
            post_ndcg_5s.append(ndcg_at_k(post_ids, rel_ids, k=5))

            # Find best rank in pre vs post
            pre_best_rank = next((i for i, cid in enumerate(pre_ids, start=1) if cid in rel_ids), 999)
            post_best_rank = next((i for i, cid in enumerate(post_ids, start=1) if cid in rel_ids), 999)

            if post_best_rank < pre_best_rank:
                promotions += 1

            if pre_best_rank < 999 and post_best_rank < 999:
                # Positive rank shift = chunk moved up towards rank 1
                rank_shifts.append(float(pre_best_rank - post_best_rank))

        n_q = float(len(evaluable_queries))
        mean_pre_mrr = sum(pre_mrrs) / n_q
        mean_post_mrr = sum(post_mrrs) / n_q
        mean_pre_hit1 = sum(pre_hit_1s) / n_q
        mean_post_hit1 = sum(post_hit_1s) / n_q
        mean_pre_ndcg5 = sum(pre_ndcg_5s) / n_q
        mean_post_ndcg5 = sum(post_ndcg_5s) / n_q

        return RerankerMetrics(
            pre_rerank_mrr=mean_pre_mrr,
            post_rerank_mrr=mean_post_mrr,
            delta_mrr=mean_post_mrr - mean_pre_mrr,
            pre_rerank_hit_1=mean_pre_hit1,
            post_rerank_hit_1=mean_post_hit1,
            delta_hit_1=mean_post_hit1 - mean_pre_hit1,
            pre_rerank_ndcg_5=mean_pre_ndcg5,
            post_rerank_ndcg_5=mean_post_ndcg5,
            delta_ndcg_5=mean_post_ndcg5 - mean_pre_ndcg5,
            promotion_rate=float(promotions) / n_q,
            average_rank_shift=sum(rank_shifts) / float(len(rank_shifts)) if rank_shifts else 0.0,
        )
