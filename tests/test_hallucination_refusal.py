import pytest
from src.agent.agent import LocalQwenAgent
from src.agent.llm import MockOllamaClient
from src.agent.tools import HybridSearchTool, RerankTool
from src.retrieval.models import SearchResult
from src.reranking.models import RerankedResult


def test_grounded_refusal_when_retrieval_finds_no_relevant_docs():
    """Verify system gracefully refuses and sets refusal=True when no relevant docs exist."""
    class EmptyRetriever:
        def hybrid_search(self, query: str, top_k: int = 20, filters=None):
            return []

    class EmptyReranker:
        def rerank(self, query: str, candidates, top_k: int = 5):
            return []

    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(default_response="GRADE: BAD\nMISSING: all\nREASON: Outside corpus"),
        hybrid_search_tool=HybridSearchTool(retriever=EmptyRetriever()),
        rerank_tool=RerankTool(reranker=EmptyReranker()),
    )

    out_of_domain_query = "What were the quarterly revenue earnings of Tesla in Q3 2025?"
    response = agent.run(out_of_domain_query)

    assert "insufficient evidence" in response.answer.lower() or "insufficient information" in response.answer.lower()
    assert response.final_evidence_grade == "BAD"
    assert response.metadata.get("refusal") is True


def test_grounded_refusal_when_evidence_is_irrelevant_or_below_threshold():
    """Verify system refuses when candidate chunks are irrelevant (cross-encoder score below min threshold)."""
    irrelevant_chunk = SearchResult(
        chunk_id="chunk_unrelated",
        text="Recipe for baking chocolate chip sourdough bread.",
        source="baking.html",
        metadata={"filename": "baking.html"},
        score=0.001,
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.001,
    )

    class IrrelevantRetriever:
        def hybrid_search(self, query: str, top_k: int = 20, filters=None):
            return [irrelevant_chunk]

    class VeryLowScoreReranker:
        def rerank(self, query: str, candidates, top_k: int = 5):
            # Score far below CRAG_MIN_RERANK_SCORE (-5.0)
            return [
                RerankedResult(
                    chunk_id=irrelevant_chunk.chunk_id,
                    text=irrelevant_chunk.text,
                    source=irrelevant_chunk.source,
                    metadata=irrelevant_chunk.metadata,
                    score=-9.8,
                    dense_rank=1,
                    sparse_rank=1,
                    rrf_score=0.001,
                    rerank_score=-9.8,
                    rerank_rank=1,
                )
            ]

    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(default_response="GRADE: BAD\nMISSING: all\nREASON: Irrelevant context"),
        hybrid_search_tool=HybridSearchTool(retriever=IrrelevantRetriever()),
        rerank_tool=RerankTool(reranker=VeryLowScoreReranker()),
    )

    response = agent.run("What is the quantum superposition principle in physics?")
    assert "insufficient evidence" in response.answer.lower()
    assert response.final_evidence_grade == "BAD"
    assert response.metadata.get("refusal") is True
