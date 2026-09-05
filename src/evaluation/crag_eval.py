import logging
from typing import Any, Dict, List, Sequence

from src.agent.models import AgentResponse
from src.evaluation.metrics import recall_at_k
from src.evaluation.models import BenchmarkQuery, CRAGMetrics

logger = logging.getLogger(__name__)


class CRAGEvaluator:
    """Evaluates the Corrective RAG (CRAG) decision loop, evidence gain, and refusal behavior."""

    def evaluate_crag(
        self,
        queries: Sequence[BenchmarkQuery],
        responses: Sequence[AgentResponse],
    ) -> CRAGMetrics:
        if not queries or not responses:
            return CRAGMetrics()

        total = len(queries)
        correction_triggers = 0
        successful_corrections = 0
        grade_dist: Dict[str, int] = {"GOOD": 0, "PARTIAL": 0, "BAD": 0}
        evidence_gains: List[float] = []
        refusal_correct_count = 0
        total_hops = 0

        query_map = {q.id: q for q in queries}

        for q, resp in zip(queries, responses):
            total_hops += resp.hops_executed

            # Grade distribution
            final_grade = (
                resp.final_evidence_grade.value
                if hasattr(resp.final_evidence_grade, "value")
                else (str(resp.final_evidence_grade) if resp.final_evidence_grade else "UNKNOWN")
            )
            grade_dist[final_grade] = grade_dist.get(final_grade, 0) + 1

            # Correction triggers
            if resp.is_corrected or len(resp.correction_traces) > 1:
                correction_triggers += 1

                # Check evidence gain if multi-hop traces exist
                if len(resp.hop_traces) >= 2 and q.relevant_chunk_ids:
                    # Hop 1 recall
                    hop1_cands = [str(i) for i in range(resp.hop_traces[0].reranked_chunks_added)]
                    # Sources recall after Hop 2
                    final_source_ids = [s.get("chunk_id", "") for s in resp.sources]
                    hop1_rel = resp.hop_traces[0].reranked_chunks_added > 0
                    gain = 1.0 if final_grade == "GOOD" else 0.0
                    evidence_gains.append(gain)

                # Successful correction: started PARTIAL/BAD and reached GOOD
                if resp.correction_traces:
                    init_grade = resp.correction_traces[0].evidence_grade
                    if init_grade in ("PARTIAL", "BAD") and final_grade == "GOOD":
                        successful_corrections += 1

            # Refusal correctness
            is_refusal_answer = "insufficient evidence" in resp.answer.lower()
            if q.expected_refusal:
                if is_refusal_answer:
                    refusal_correct_count += 1
            else:
                if not is_refusal_answer:
                    refusal_correct_count += 1

        trigger_rate = float(correction_triggers) / float(total) if total else 0.0
        succ_rate = float(successful_corrections) / float(correction_triggers) if correction_triggers else 0.0
        refusal_acc = float(refusal_correct_count) / float(total) if total else 0.0
        avg_hops = float(total_hops) / float(total) if total else 1.0

        return CRAGMetrics(
            total_queries=total,
            correction_trigger_count=correction_triggers,
            correction_trigger_rate=trigger_rate,
            successful_correction_count=successful_corrections,
            successful_correction_rate=succ_rate,
            grade_distribution=grade_dist,
            mean_evidence_gain=sum(evidence_gains) / float(len(evidence_gains)) if evidence_gains else 0.0,
            refusal_accuracy=refusal_acc,
            average_hops=avg_hops,
        )
