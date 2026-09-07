import pytest
from src.agent.agent import LocalQwenAgent
from src.agent.llm import MockOllamaClient
from src.agent.prompts import SYNTHESIS_SYSTEM_PROMPT, build_synthesis_prompt, format_context_blocks
from src.agent.tools import HybridSearchTool, RerankTool
from src.reranking.models import RerankedResult
from src.retrieval.models import SearchResult


def _make_reranked(chunk_id: str, text: str, source: str, score: float, section: str) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id,
        text=text,
        source=source,
        metadata={"filename": source, "section": section},
        score=score,
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.03,
        rerank_score=score,
        rerank_rank=1,
    )


def test_synthesis_prompt_includes_multi_doc_guidelines():
    """Verify system prompt explicitly instructs coherent cross-document synthesis."""
    assert "Multi-Document Synthesis" in SYNTHESIS_SYSTEM_PROMPT
    assert "synthesize the facts coherently across the sources" in SYNTHESIS_SYSTEM_PROMPT
    assert "unsupported inferential leaps" in SYNTHESIS_SYSTEM_PROMPT


def test_format_context_blocks_preserves_distinct_sources():
    """Verify format_context_blocks clearly demarcates multiple distinct document sources."""
    chunk_a = _make_reranked(
        "doc_a_chunk_1",
        "Feature X was introduced in Kubernetes version 1.28.",
        "kubernetes-docs.html",
        4.2,
        "Release Notes",
    )
    chunk_b = _make_reranked(
        "doc_b_chunk_1",
        "Kubernetes version 1.28 requires configuration flag --enable-feature-x=true.",
        "cluster-config.html",
        3.9,
        "Configuration Guide",
    )

    blocks = format_context_blocks([chunk_a, chunk_b])

    assert "kubernetes-docs.html" in blocks
    assert "cluster-config.html" in blocks
    assert "Feature X was introduced in Kubernetes version 1.28." in blocks
    assert "requires configuration flag --enable-feature-x=true." in blocks


def test_deterministic_multi_document_synthesis():
    """Deterministic test proving cross-document synthesis combining facts from 2 documents."""
    chunk_a = _make_reranked(
        "chunk_docker",
        "Docker user-defined bridge networks provide automatic DNS resolution between containers on the same bridge.",
        "docker-bridge.html",
        4.5,
        "Docker Bridge Networking",
    )
    chunk_b = _make_reranked(
        "chunk_k8s",
        "In Kubernetes, every Pod gets its own cluster-wide IP address, enabling pod-to-pod communication without port forwarding.",
        "kubernetes-pods.html",
        4.3,
        "Pod Networking Model",
    )

    class MockMultiDocRetriever:
        def hybrid_search(self, query: str, top_k: int = 20, filters=None):
            return [
                SearchResult(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    metadata=c.metadata,
                    score=c.score,
                    dense_rank=1,
                    sparse_rank=1,
                    rrf_score=0.03,
                )
                for c in [chunk_a, chunk_b]
            ]

    class MockMultiDocReranker:
        def rerank(self, query: str, candidates, top_k: int = 5):
            return [chunk_a, chunk_b]

    captured_prompts = []

    def multi_doc_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        if "User Question:" in prompt or "Instructions:" in prompt:
            # Grounded answer synthesizing both sources
            return (
                "Based on the documentation, Docker user-defined bridge networks provide automatic DNS resolution "
                "between containers (from docker-bridge.html), whereas the Kubernetes networking model allocates "
                "every Pod its own unique cluster-wide IP address without needing port forwarding (from kubernetes-pods.html)."
            )
        if "Evaluate evidence quality:" in prompt:
            return "GRADE: GOOD\nMISSING: NONE\nREASON: Both Docker and Kubernetes networking context present."
        return "RETRIEVE"

    search_tool = HybridSearchTool(retriever=MockMultiDocRetriever())
    rerank_tool = RerankTool(reranker=MockMultiDocReranker())
    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(response_generator=multi_doc_llm),
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    query = "How do Docker bridge networks and the Kubernetes Pod network model compare in how containers communicate?"
    response = agent.run(query)

    # 1. Both documents must be attributed in sources
    source_files = {s["source"] for s in response.sources}
    assert "docker-bridge.html" in source_files
    assert "kubernetes-pods.html" in source_files
    assert len(response.sources) == 2

    # 2. Final synthesized answer contains facts from BOTH documents
    assert "Docker user-defined bridge" in response.answer or "DNS resolution" in response.answer
    assert "Kubernetes" in response.answer or "cluster-wide IP" in response.answer

    # 3. Context supplied to LLM synthesis prompt contained both chunks
    synthesis_prompts = [p for p in captured_prompts if "User Question:" in p]
    assert len(synthesis_prompts) > 0
    assert "docker-bridge.html" in synthesis_prompts[0]
    assert "kubernetes-pods.html" in synthesis_prompts[0]
