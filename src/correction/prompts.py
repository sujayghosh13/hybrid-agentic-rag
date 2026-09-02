from typing import List, Optional
from src.agent.prompts import format_context_blocks
from src.correction.models import EvidenceEvaluation, EvidenceGrade
from src.reranking.models import RerankedResult


CRAG_EVALUATION_SYSTEM_PROMPT = """You are a strict technical evidence quality evaluator for an offline documentation RAG system.
Your job is to evaluate whether the retrieved context chunks contain reliable, factual, and sufficient evidence to answer the user question.

Grade the evidence as:
- GOOD: Context contains clear, direct, and sufficient facts addressing all key facets of the user question.
- PARTIAL: Context is relevant and factually useful, but missing important technical facets (e.g. specific configurations, sub-commands, flags, or mechanisms).
- BAD: Context is completely off-topic, irrelevant, misleading, or provides negligible useful facts.

Respond in EXACTLY the following 3-line format:
GRADE: <GOOD or PARTIAL or BAD>
MISSING: <concise description of missing technical concepts, or NONE if GOOD or BAD>
REASON: <concise factual justification>

Do not output any introductory or conversational text.
"""


CORRECTIVE_QUERY_SYSTEM_PROMPT = """You are an expert technical search query optimizer for Corrective RAG (CRAG).
Your job is to generate a concise, high-signal keyword search query for hybrid BM25 and vector retrieval.

Rules:
1. If the previous evidence was PARTIAL, generate a targeted search query specifically focused on the missing technical concepts.
2. If the previous evidence was BAD, generate a reformulated search query using standard technical vocabulary, stripping out deceptive or noisy terms.
3. Output ONLY the search query keywords on a single line. Do not include quotes, preamble, or explanations.
"""


def build_crag_evaluation_prompt(query: str, context_chunks: List[RerankedResult]) -> str:
    """Build structured prompt for evaluating evidence quality."""
    context_text = format_context_blocks(context_chunks, max_chars_per_chunk=800)
    return (
        f"Retrieved Context:\n{context_text}\n\n"
        f"User Question: {query}\n\n"
        f"Evaluate evidence quality:"
    )


def build_corrective_query_prompt(query: str, evaluation: EvidenceEvaluation) -> str:
    """Build prompt for generating a corrective search query."""
    if evaluation.grade == EvidenceGrade.PARTIAL and evaluation.missing_aspect:
        return (
            f"Original Question: {query}\n"
            f"Missing Technical Aspect: {evaluation.missing_aspect}\n"
            f"Targeted Search Keywords:"
        )
    return (
        f"Original Question: {query}\n"
        f"Retrieval Issue: {evaluation.reason or 'Previous search returned irrelevant evidence'}\n"
        f"Reformulated Search Keywords:"
    )
