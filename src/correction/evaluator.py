import logging
from typing import List, Optional

from src.agent.llm import BaseLLMClient, OllamaClient
from src.config import settings
from src.correction.models import EvidenceEvaluation, EvidenceGrade
from src.correction.prompts import CRAG_EVALUATION_SYSTEM_PROMPT, build_crag_evaluation_prompt
from src.reranking.models import RerankedResult

logger = logging.getLogger(__name__)


class EvidenceEvaluator:
    """Evaluates the quality and sufficiency of retrieved evidence in Corrective RAG."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm = llm_client or OllamaClient()

    def evaluate(
        self,
        query: str,
        context_chunks: List[RerankedResult],
    ) -> EvidenceEvaluation:
        """Evaluate evidence quality with deterministic fast-paths followed by LLM evaluation."""
        # Tier 1: Deterministic checks (0 LLM cost)
        if not context_chunks:
            logger.info("[Evidence Evaluator] Empty context chunks -> Deterministic BAD.")
            return EvidenceEvaluation(
                grade=EvidenceGrade.BAD,
                missing_aspect="All technical details",
                reason="No candidate chunks retrieved.",
            )

        top_score = max(chunk.rerank_score for chunk in context_chunks)
        min_score_threshold = getattr(settings, "crag_min_rerank_score", -5.0)
        if top_score < min_score_threshold:
            logger.info(f"[Evidence Evaluator] Top score {top_score:.3f} < {min_score_threshold} -> Deterministic BAD.")
            return EvidenceEvaluation(
                grade=EvidenceGrade.BAD,
                missing_aspect="Relevant technical context",
                reason=f"Top rerank score ({top_score:.3f}) below relevance threshold.",
            )

        if not getattr(settings, "crag_enabled", True):
            return EvidenceEvaluation(grade=EvidenceGrade.GOOD, reason="CRAG disabled.")

        # Tier 2: Structured LLM Grader
        prompt = build_crag_evaluation_prompt(query, context_chunks)
        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system=CRAG_EVALUATION_SYSTEM_PROMPT,
                temperature=0.0,
            )
            return self._parse_evaluation_response(raw_response)
        except Exception as e:
            logger.warning(f"EvidenceEvaluator encountered error: {e}. Defaulting to GOOD based on rerank score.")
            return EvidenceEvaluation(
                grade=EvidenceGrade.GOOD,
                reason=f"Evaluator error fallback ({e}).",
            )

    def _parse_evaluation_response(self, text: str) -> EvidenceEvaluation:
        """Parse structured 3-line evaluator output with backward-compatible format support."""
        grade = EvidenceGrade.GOOD
        missing_aspect: Optional[str] = None
        reason: Optional[str] = None

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        for line in lines:
            upper = line.upper()
            if upper.startswith("GRADE:"):
                val = line.split(":", 1)[1].strip().upper()
                if "PARTIAL" in val:
                    grade = EvidenceGrade.PARTIAL
                elif "BAD" in val:
                    grade = EvidenceGrade.BAD
                elif "GOOD" in val:
                    grade = EvidenceGrade.GOOD
            elif upper.startswith("MISSING:"):
                missing = line.split(":", 1)[1].strip()
                if missing and missing.upper() != "NONE":
                    missing_aspect = missing
            elif upper.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        # Backward compatibility for legacy Phase 4B prompt outputs (e.g. "INSUFFICIENT: ...", "SUFFICIENT")
        upper_text = text.upper()
        if "INSUFFICIENT" in upper_text and grade == EvidenceGrade.GOOD:
            grade = EvidenceGrade.PARTIAL
            if not missing_aspect:
                for line in lines:
                    if line.upper().startswith("INSUFFICIENT"):
                        parts = line.split(":", 1)
                        if len(parts) > 1 and parts[1].strip():
                            missing_aspect = parts[1].strip()
                            break

        # Sanity check consistency
        if grade == EvidenceGrade.PARTIAL and not missing_aspect:
            missing_aspect = "additional technical details needed"

        logger.info(f"[Evidence Evaluator] Result: Grade={grade.value} | Missing={missing_aspect} | Reason={reason}")
        return EvidenceEvaluation(
            grade=grade,
            missing_aspect=missing_aspect,
            reason=reason,
        )
