import logging
from typing import Optional

from src.agent.llm import BaseLLMClient, OllamaClient, call_llm_generate
from src.correction.models import EvidenceEvaluation, EvidenceGrade
from src.correction.prompts import (
    CORRECTIVE_QUERY_SYSTEM_PROMPT,
    build_corrective_query_prompt,
)

logger = logging.getLogger(__name__)


class CorrectiveActionEngine:
    """Generates targeted or reformulated corrective retrieval queries in CRAG."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm = llm_client or OllamaClient()

    def generate_corrective_query(
        self,
        query: str,
        evaluation: EvidenceEvaluation,
    ) -> str:
        """Generate a corrective query tailored to PARTIAL or BAD evidence grades."""
        if evaluation.grade == EvidenceGrade.GOOD:
            return query

        prompt = build_corrective_query_prompt(query, evaluation)
        try:
            raw_response = call_llm_generate(
                self.llm,
                prompt=prompt,
                system=CORRECTIVE_QUERY_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=40,
                stop=["\n"],
            )
            cleaned = raw_response.strip().strip('"').strip("'").split("\n")[0].strip()
            if cleaned and len(cleaned) > 2:
                logger.info(f"[Corrective Action] Grade={evaluation.grade.value} | Corrective Query: '{cleaned}'")
                return cleaned
        except Exception as e:
            logger.warning(f"CorrectiveActionEngine encountered error: {e}. Falling back to original query.")

        # Fallback query if generation fails
        if evaluation.grade == EvidenceGrade.PARTIAL and evaluation.missing_aspect:
            return f"{query} {evaluation.missing_aspect}"
        return query
