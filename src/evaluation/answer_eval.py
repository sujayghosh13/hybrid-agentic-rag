import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from src.agent.models import AgentResponse
from src.evaluation.metrics import aspect_coverage_score, source_attribution_coverage
from src.evaluation.models import AnswerMetrics, BenchmarkQuery

logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """Evaluates the synthesized final answers, aspect coverage, source attribution, and refusal accuracy."""

    def __init__(self, corpus_chunk_ids: Optional[Set[str]] = None):
        self.corpus_chunk_ids = corpus_chunk_ids or set()

    def evaluate_answers(
        self,
        queries: Sequence[BenchmarkQuery],
        responses: Sequence[AgentResponse],
    ) -> AnswerMetrics:
        if not queries or not responses:
            return AnswerMetrics()

        aspect_coverages: List[float] = []
        refusal_correct = 0
        source_coverages: List[float] = []

        for q, resp in zip(queries, responses):
            # 1. Aspect coverage
            if not q.expected_refusal:
                cov = aspect_coverage_score(resp.answer, q.expected_aspects)
                aspect_coverages.append(cov)

            # 2. Refusal accuracy
            is_refusal = "insufficient evidence" in resp.answer.lower()
            if q.expected_refusal and is_refusal:
                refusal_correct += 1
            elif not q.expected_refusal and not is_refusal:
                refusal_correct += 1

            # 3. Source attribution coverage
            src_cov = source_attribution_coverage(resp.sources, self.corpus_chunk_ids)
            source_coverages.append(src_cov)

        n_non_refusal = len(aspect_coverages)
        mean_aspect = sum(aspect_coverages) / float(n_non_refusal) if n_non_refusal else 1.0
        ref_acc = float(refusal_correct) / float(len(queries)) if queries else 1.0
        mean_src_cov = sum(source_coverages) / float(len(source_coverages)) if source_coverages else 1.0

        return AnswerMetrics(
            mean_aspect_coverage=mean_aspect,
            refusal_accuracy=ref_acc,
            mean_source_attribution_coverage=mean_src_cov,
            mean_faithfulness=None,
            mean_relevance=None,
        )
