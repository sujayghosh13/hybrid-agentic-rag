import pytest

from src.agent.agent import LocalQwenAgent
from src.agent.llm import MockOllamaClient, OllamaConnectionError
from src.agent.models import AgentResponse, HopTrace, ToolCall
from src.agent.tools import BaseTool, HybridSearchTool, RerankTool, ToolRegistry
from src.config import settings
from src.reranking.models import RerankedResult
from src.retrieval.models import SearchResult


class MockHybridRetriever:
    """Mock HybridRetriever returning candidate search results based on query."""

    def __init__(self, query_responses=None):
        self.query_responses = query_responses or {}

    def hybrid_search(self, query: str, top_k: int = 20):
        if query in self.query_responses:
            return self.query_responses[query]

        return [
            SearchResult(
                chunk_id="chunk_1",
                text="Docker bridge network enables communication between containers on the same host.",
                source="data/raw/docker-bridge.html",
                metadata={"filename": "docker-bridge.html", "section": "Bridge Driver"},
                score=0.032,
                dense_rank=1,
                sparse_rank=1,
                rrf_score=0.032,
            ),
            SearchResult(
                chunk_id="chunk_2",
                text="User-defined bridge networks provide automatic DNS resolution between containers.",
                source="data/raw/docker-networking.html",
                metadata={"filename": "docker-networking.html", "section": "User-Defined Bridge"},
                score=0.025,
                dense_rank=2,
                sparse_rank=3,
                rrf_score=0.025,
            ),
        ]


class MockCrossEncoderReranker:
    """Mock CrossEncoderReranker returning scored reranked results."""

    def rerank(self, query: str, candidates, top_k: int = 5):
        reranked = []
        for rank, cand in enumerate(candidates, start=1):
            reranked.append(
                RerankedResult(
                    chunk_id=cand.chunk_id,
                    text=cand.text,
                    source=cand.source,
                    metadata=cand.metadata,
                    score=3.5 - (rank * 0.5),
                    dense_rank=cand.dense_rank,
                    sparse_rank=cand.sparse_rank,
                    rrf_score=cand.rrf_score,
                    rerank_score=3.5 - (rank * 0.5),
                    rerank_rank=rank,
                )
            )
        return reranked[:top_k]


@pytest.fixture
def mock_tools():
    search_tool = HybridSearchTool(retriever=MockHybridRetriever())
    rerank_tool = RerankTool(reranker=MockCrossEncoderReranker())
    return search_tool, rerank_tool


def test_agent_initialization(mock_tools):
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient()
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    assert agent.llm == mock_llm
    assert agent.search_tool == search_tool
    assert agent.rerank_tool == rerank_tool
    assert agent.registry.get("hybrid_search") is not None
    assert agent.registry.get("rerank") is not None


def test_tool_execution(mock_tools):
    search_tool, rerank_tool = mock_tools

    # Test hybrid search tool
    candidates = search_tool.execute(query="bridge network", top_k=2)
    assert len(candidates) == 2
    assert candidates[0].chunk_id == "chunk_1"
    assert candidates[0].rrf_score == 0.032

    # Test rerank tool
    reranked = rerank_tool.execute(query="bridge network", candidates=candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "chunk_1"
    assert reranked[0].rerank_rank == 1
    assert reranked[0].dense_rank == 1
    assert reranked[0].sparse_rank == 1


def test_agent_routing_decision(mock_tools):
    search_tool, rerank_tool = mock_tools

    # 1. Routing for technical question
    mock_llm_retrieve = MockOllamaClient(default_response="RETRIEVE")
    agent_retrieve = LocalQwenAgent(
        llm_client=mock_llm_retrieve,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )
    assert agent_retrieve.should_retrieve("How does Docker bridge networking work?") is True

    # 2. Routing for conversational greeting
    assert agent_retrieve.should_retrieve("Hello") is False


def test_agent_full_retrieval_and_answer_synthesis(mock_tools):
    search_tool, rerank_tool = mock_tools

    def mock_response(prompt: str) -> str:
        if "Decision:" in prompt:
            return "RETRIEVE"
        if "Optimized Search Keywords:" in prompt:
            return "Docker bridge networking configuration"
        if "Is the context sufficient" in prompt:
            return "SUFFICIENT"
        return "Docker bridge networking creates a private internal network on the host."

    mock_llm = MockOllamaClient(response_generator=mock_response)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    query = "How does Docker bridge networking work?"
    response = agent.run(query)

    assert isinstance(response, AgentResponse)
    assert response.query == query
    assert "Docker bridge networking" in response.answer
    assert response.retrieval_needed is True
    assert len(response.sources) == 2
    assert response.sources[0]["chunk_id"] == "chunk_1"
    assert response.sources[0]["rerank_score"] == 3.0
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].tool_name == "hybrid_search"
    assert response.tool_calls[1].tool_name == "rerank"
    assert len(response.thought_process) > 0
    assert len(response.hop_traces) == 1
    assert response.hop_traces[0].is_sufficient is True


def test_agent_direct_conversational_query(mock_tools):
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient(default_response="Hello! I am your technical documentation assistant.")
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("Hello")
    assert response.retrieval_needed is False
    assert len(response.sources) == 0
    assert len(response.tool_calls) == 0
    assert "Hello!" in response.answer


def test_agent_empty_query(mock_tools):
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient()
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("")
    assert response.retrieval_needed is False
    assert "valid question" in response.answer


def test_ollama_connection_error_handling(mock_tools):
    search_tool, rerank_tool = mock_tools

    class FailingLLMClient(MockOllamaClient):
        def generate(self, prompt, system=None, temperature=None):
            if "Decision:" in prompt:
                return "RETRIEVE"
            if "Optimized Search Keywords:" in prompt:
                return "Docker networking"
            if "Is the context sufficient" in prompt:
                return "SUFFICIENT"
            raise OllamaConnectionError("Connection refused on port 11434")

    failing_llm = FailingLLMClient()
    agent = LocalQwenAgent(
        llm_client=failing_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("How does Docker networking work?")
    assert "Could not connect to the local Ollama LLM server" in response.answer
    assert response.metadata["ollama_status"] == "unreachable"
    assert len(response.sources) == 2


# =========================================================================
# Phase 4B Unit Tests: Query Rewriting, Sufficiency, and Multi-Hop Loop
# =========================================================================


def test_query_rewriting_hop1(mock_tools):
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient(default_response="Docker bridge network driver configuration")
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    rewritten = agent.rewrite_query("Can you tell me how to configure the bridge driver in Docker?")
    assert rewritten == "Docker bridge network driver configuration"


def test_query_rewriting_with_missing_aspect(mock_tools):
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient(default_response="Docker bridge network DNS resolution automatic")
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    rewritten = agent.rewrite_query(
        "How does Docker bridge networking work?",
        missing_aspect="automatic container DNS resolution",
    )
    assert rewritten == "Docker bridge network DNS resolution automatic"


def test_sufficiency_check_sufficient(mock_tools):
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient(default_response="SUFFICIENT")
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    candidates = search_tool.execute("bridge network")
    reranked = rerank_tool.execute("bridge network", candidates)
    is_sufficient, missing = agent.evaluate_sufficiency("How does Docker bridge work?", reranked)

    assert is_sufficient is True
    assert missing is None


def test_sufficiency_check_insufficient(mock_tools):
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient(default_response="INSUFFICIENT: port publishing and IP masquerading")
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    candidates = search_tool.execute("bridge network")
    reranked = rerank_tool.execute("bridge network", candidates)
    is_sufficient, missing = agent.evaluate_sufficiency("How does Docker bridge work?", reranked)

    assert is_sufficient is False
    assert missing == "port publishing and IP masquerading"


def test_multi_hop_retrieval_and_deduplication():
    # Hop 1 returns chunk_1 and chunk_2
    hop1_candidates = [
        SearchResult(
            chunk_id="chunk_1",
            text="Docker bridge network basics.",
            source="docker-bridge.html",
            metadata={"filename": "docker-bridge.html"},
            score=0.03,
            dense_rank=1,
            sparse_rank=1,
            rrf_score=0.03,
        ),
        SearchResult(
            chunk_id="chunk_2",
            text="User-defined bridge networks.",
            source="docker-networking.html",
            metadata={"filename": "docker-networking.html"},
            score=0.02,
            dense_rank=2,
            sparse_rank=2,
            rrf_score=0.02,
        ),
    ]

    # Hop 2 returns chunk_2 (duplicate) and chunk_3 (new)
    hop2_candidates = [
        SearchResult(
            chunk_id="chunk_2",
            text="User-defined bridge networks updated.",
            source="docker-networking.html",
            metadata={"filename": "docker-networking.html"},
            score=0.02,
            dense_rank=2,
            sparse_rank=2,
            rrf_score=0.02,
        ),
        SearchResult(
            chunk_id="chunk_3",
            text="Port publishing and iptables rules on Docker bridge.",
            source="docker-ports.html",
            metadata={"filename": "docker-ports.html"},
            score=0.035,
            dense_rank=1,
            sparse_rank=1,
            rrf_score=0.035,
        ),
    ]

    mock_retriever = MockHybridRetriever(
        query_responses={
            "Docker bridge network basics": hop1_candidates,
            "Docker bridge port publishing": hop2_candidates,
        }
    )
    search_tool = HybridSearchTool(retriever=mock_retriever)
    rerank_tool = RerankTool(reranker=MockCrossEncoderReranker())

    call_count = {"sufficiency": 0, "rewrite": 0}

    def mock_llm_logic(prompt: str) -> str:
        if "Optimized Search Keywords:" in prompt:
            return "Docker bridge network basics"
        if "Generate a targeted search query" in prompt or "Missing Technical Aspect:" in prompt or "Targeted Search Keywords:" in prompt:
            return "Docker bridge port publishing"
        if "Is the context sufficient" in prompt or "Evaluate evidence quality:" in prompt:
            call_count["sufficiency"] += 1
            if call_count["sufficiency"] == 1:
                return "INSUFFICIENT: port publishing"
            return "SUFFICIENT"
        return "Comprehensive answer covering bridge networking and port publishing."

    mock_llm = MockOllamaClient(response_generator=mock_llm_logic)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("How does Docker bridge networking and port publishing work?")

    assert response.hops_executed == 2
    assert len(response.hop_traces) == 2
    assert response.hop_traces[0].is_sufficient is False
    assert response.hop_traces[1].is_sufficient is True

    # Confirm unique deduplicated chunks: chunk_1, chunk_2, chunk_3
    chunk_ids = [s["chunk_id"] for s in response.sources]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert "chunk_1" in chunk_ids
    assert "chunk_2" in chunk_ids
    assert "chunk_3" in chunk_ids


def test_max_hops_safety_limit(mock_tools):
    search_tool, rerank_tool = mock_tools
    rewrite_count = {"count": 0}

    def mock_always_insufficient(prompt: str) -> str:
        if "Optimized Search Keywords:" in prompt:
            return "Docker bridge query 1"
        if "Generate a targeted search query" in prompt or "Missing Technical Aspect:" in prompt or "Targeted Search Keywords:" in prompt:
            rewrite_count["count"] += 1
            return f"Docker bridge query {rewrite_count['count'] + 1}"
        if "Is the context sufficient" in prompt or "Evaluate evidence quality:" in prompt:
            return "INSUFFICIENT: still missing complex details"
        return "Answer synthesized on available context despite partial sufficiency."

    mock_llm = MockOllamaClient(response_generator=mock_always_insufficient)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("How does Docker bridge networking work in all complex edge cases?")

    # Max hops is strictly bounded at 2
    assert response.hops_executed == 2
    assert len(response.hop_traces) == 2
    assert response.hop_traces[0].is_sufficient is False
    assert response.hop_traces[1].is_sufficient is False
    assert len(response.sources) > 0


def test_duplicate_query_loop_break(mock_tools):
    search_tool, rerank_tool = mock_tools

    def mock_identical_rewrite(prompt: str) -> str:
        if "Optimized Search Keywords:" in prompt or "Generate a targeted search query" in prompt or "Missing Technical Aspect:" in prompt:
            return "Identical Docker Query"
        if "Is the context sufficient" in prompt or "Evaluate evidence quality:" in prompt:
            return "INSUFFICIENT: missing info"
        return "Answer."

    mock_llm = MockOllamaClient(response_generator=mock_identical_rewrite)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("Tell me about Docker bridge networking.")

    # Second hop generates same query -> breaks loop immediately
    assert response.hops_executed <= 2
    assert "was already searched" in " ".join(response.thought_process) or "already searched" in " ".join(response.thought_process)


# =========================================================================
# Phase 5 Unit Tests: Corrective RAG (CRAG) & Global Retrieval Budget
# =========================================================================
from src.correction.evaluator import EvidenceEvaluator
from src.correction.models import EvidenceGrade


def test_crag_good_path_no_correction(mock_tools):
    search_tool, rerank_tool = mock_tools

    def mock_logic(prompt: str) -> str:
        if "Evaluate evidence quality:" in prompt:
            return "GRADE: GOOD\nMISSING: NONE\nREASON: Complete facts present."
        return "Complete grounded answer about Docker bridge."

    mock_llm = MockOllamaClient(response_generator=mock_logic)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("How does Docker bridge networking work?")

    assert response.hops_executed == 1
    assert response.final_evidence_grade == "GOOD"
    assert response.is_corrected is False
    assert len(response.correction_traces) == 1
    assert response.correction_traces[0].action_taken == "PROCEED_TO_SYNTHESIS"
    assert "Complete grounded answer" in response.answer


def test_crag_partial_triggers_single_corrective_hop():
    hop1_candidates = [
        SearchResult(
            chunk_id="chunk_1",
            text="Docker bridge basics.",
            source="docker-bridge.html",
            metadata={"filename": "docker-bridge.html"},
            score=0.03,
            dense_rank=1,
            sparse_rank=1,
            rrf_score=0.03,
        ),
    ]
    hop2_candidates = [
        SearchResult(
            chunk_id="chunk_2",
            text="Docker port publishing and masquerading.",
            source="docker-bridge.html",
            metadata={"filename": "docker-bridge.html"},
            score=0.04,
            dense_rank=1,
            sparse_rank=1,
            rrf_score=0.04,
        ),
    ]

    mock_retriever = MockHybridRetriever(
        query_responses={
            "How does Docker bridge networking and port publishing work?": hop1_candidates,
            "Docker bridge port publishing": hop2_candidates,
        }
    )
    search_tool = HybridSearchTool(retriever=mock_retriever)
    rerank_tool = RerankTool(reranker=MockCrossEncoderReranker())

    eval_count = {"count": 0}

    def mock_crag_flow(prompt: str) -> str:
        if "Missing Technical Aspect:" in prompt or "Targeted Search Keywords:" in prompt:
            return "Docker bridge port publishing"
        if "Evaluate evidence quality:" in prompt:
            eval_count["count"] += 1
            if eval_count["count"] == 1:
                return "GRADE: PARTIAL\nMISSING: port publishing\nREASON: Missing port forwarding facts."
            return "GRADE: GOOD\nMISSING: NONE\nREASON: All port facts now present."
        return "Comprehensive answer covering bridge and port publishing."

    mock_llm = MockOllamaClient(response_generator=mock_crag_flow)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("How does Docker bridge networking and port publishing work?")

    assert response.hops_executed == 2
    assert response.is_corrected is True
    assert response.final_evidence_grade == "GOOD"
    assert len(response.correction_traces) == 2
    assert response.correction_traces[0].evidence_grade == EvidenceGrade.PARTIAL
    assert response.correction_traces[1].evidence_grade == EvidenceGrade.GOOD

    # Deduplicated chunks: chunk_1 and chunk_2
    source_ids = [s["chunk_id"] for s in response.sources]
    assert "chunk_1" in source_ids
    assert "chunk_2" in source_ids


def test_crag_global_retrieval_budget_never_exceeds_two(mock_tools):
    search_tool, rerank_tool = mock_tools
    eval_count = {"count": 0}

    def mock_persistent_partial(prompt: str) -> str:
        if "Missing Technical Aspect:" in prompt:
            return "Docker bridge obscure edge case"
        if "Evaluate evidence quality:" in prompt:
            eval_count["count"] += 1
            return "GRADE: PARTIAL\nMISSING: obscure details\nREASON: Still partial."
        return "Synthesized answer on best available evidence."

    mock_llm = MockOllamaClient(response_generator=mock_persistent_partial)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("How does Docker bridge networking work in complex edge cases?")

    # Under no circumstances can global retrieval hops exceed 2
    assert response.hops_executed <= 2
    assert len(response.hop_traces) <= 2
    assert len(response.correction_traces) <= 2


def test_crag_persistent_bad_evidence_yields_grounded_refusal():
    # Empty retriever returning no results
    mock_retriever = MockHybridRetriever(query_responses={"Docker impossible query": []})
    search_tool = HybridSearchTool(retriever=mock_retriever)
    rerank_tool = RerankTool(reranker=MockCrossEncoderReranker())

    mock_llm = MockOllamaClient(
        default_response="GRADE: BAD\nMISSING: everything\nREASON: No documentation exists."
    )
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("Docker impossible query")

    assert response.final_evidence_grade == "BAD"
    assert "insufficient evidence" in response.answer.lower()
    assert response.metadata.get("refusal") is True


def test_crag_chunk_deduplication_preserves_best_rerank_score():
    chunk_v1 = SearchResult(
        chunk_id="chunk_shared",
        text="Docker bridge text v1.",
        source="docker-bridge.html",
        metadata={"filename": "docker-bridge.html"},
        score=0.01,
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.01,
    )
    chunk_v2 = SearchResult(
        chunk_id="chunk_shared",
        text="Docker bridge text v2 with higher relevance.",
        source="docker-bridge.html",
        metadata={"filename": "docker-bridge.html"},
        score=0.05,
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.05,
    )

    mock_retriever = MockHybridRetriever(
        query_responses={
            "Docker bridge initial query": [chunk_v1],
            "Docker bridge corrective query": [chunk_v2],
        }
    )

    class DynamicReranker:
        def rerank(self, query: str, candidates, top_k: int = 5):
            score = 1.0 if "initial" in query else 4.0
            return [
                RerankedResult(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    metadata=c.metadata,
                    score=score,
                    dense_rank=1,
                    sparse_rank=1,
                    rrf_score=c.score,
                    rerank_score=score,
                    rerank_rank=1,
                )
                for c in candidates
            ]

    search_tool = HybridSearchTool(retriever=mock_retriever)
    rerank_tool = RerankTool(reranker=DynamicReranker())

    eval_count = {"count": 0}

    def mock_logic(prompt: str) -> str:
        if "Optimized Search Keywords:" in prompt:
            return "Docker bridge initial query"
        if "Missing Technical Aspect:" in prompt or "Targeted Search Keywords:" in prompt:
            return "Docker bridge corrective query"
        if "Evaluate evidence quality:" in prompt:
            eval_count["count"] += 1
            if eval_count["count"] == 1:
                return "GRADE: PARTIAL\nMISSING: better score\nREASON: partial"
            return "GRADE: GOOD\nMISSING: NONE\nREASON: good"
        return "Answer."

    mock_llm = MockOllamaClient(response_generator=mock_logic)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("Docker bridge initial query")

    assert len(response.sources) == 1
    assert response.sources[0]["chunk_id"] == "chunk_shared"
    assert response.sources[0]["rerank_score"] == 4.0  # Kept the higher score from Hop 2


def test_crag_duplicate_corrective_query_break(mock_tools):
    search_tool, rerank_tool = mock_tools

    def mock_identical_corrective(prompt: str) -> str:
        if "Optimized Search Keywords:" in prompt:
            return "Docker bridge identical query"
        if "Missing Technical Aspect:" in prompt or "Targeted Search Keywords:" in prompt:
            return "Docker bridge identical query"
        if "Evaluate evidence quality:" in prompt:
            return "GRADE: PARTIAL\nMISSING: details\nREASON: partial"
        return "Answer."

    mock_llm = MockOllamaClient(response_generator=mock_identical_corrective)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("Docker bridge identical query")

    # Corrective query matches initial query -> breaks before Hop 2 search
    assert response.hops_executed == 1
    assert "already searched" in " ".join(response.thought_process)


def test_router_and_rewriter_cache(mock_tools):
    """Verify that routing and query rewriting decisions are cached to eliminate redundant LLM calls."""
    search_tool, rerank_tool = mock_tools
    llm_call_count = {"count": 0}

    def counting_llm(prompt: str) -> str:
        llm_call_count["count"] += 1
        if "Optimized Search Keywords:" in prompt:
            return "specialized custom query"
        return "RETRIEVE"

    mock_llm = MockOllamaClient(response_generator=counting_llm)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    # First call: should query router and rewriter
    r1 = agent.should_retrieve("uncommon obscure syntax")
    q1 = agent.rewrite_query("uncommon obscure syntax")
    initial_count = llm_call_count["count"]
    assert initial_count >= 1

    # Second call with same query (or case variant): must hit cache with 0 new LLM calls
    r2 = agent.should_retrieve("UNCOMMON obscure syntax")
    q2 = agent.rewrite_query("UNCOMMON obscure syntax")
    assert r1 == r2
    assert q1 == q2
    assert llm_call_count["count"] == initial_count


def test_route_and_rewrite(mock_tools):
    """Verify unified route_and_rewrite method returns routing flag and search keywords."""
    search_tool, rerank_tool = mock_tools
    mock_llm = MockOllamaClient(default_response="docker bridge ip forwarding")
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    # Conversational query
    retrieve, search_q = agent.route_and_rewrite("hello")
    assert retrieve is False
    assert search_q == "hello"

    # Technical query
    retrieve, search_q = agent.route_and_rewrite("How does Docker bridge ip forwarding work?")
    assert retrieve is True
    assert "docker bridge" in search_q.lower()


def test_adaptive_max_tokens_calculation(mock_tools):
    """Verify adaptive max_tokens dynamically scales based on query complexity and context chunks."""
    search_tool, rerank_tool = mock_tools
    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(),
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    from src.reranking.models import RerankedResult
    dummy_chunk = RerankedResult(
        chunk_id="c1", text="Sample text", source="doc.html",
        metadata={}, score=1.0, dense_rank=1, sparse_rank=1,
        rrf_score=0.1, rerank_score=1.0, rerank_rank=1,
    )

    # Simple question with 1 chunk -> base budget 400
    t_simple = agent._calculate_adaptive_max_tokens("What is Docker?", [dummy_chunk])
    assert t_simple == 400

    # Multi-part question -> base + 150 = 550
    t_complex = agent._calculate_adaptive_max_tokens(
        "Explain how Docker bridge and host networking compare and what are the differences?",
        [dummy_chunk],
    )
    assert t_complex == 550

    # Multi-part question with 3+ chunks -> base + 150 + 100 = 650
    t_full = agent._calculate_adaptive_max_tokens(
        "Explain how Docker bridge and host networking compare?",
        [dummy_chunk, dummy_chunk, dummy_chunk],
    )
    assert t_full == 650


def test_ollama_client_num_ctx_option():
    """Verify that OllamaClient passes configured num_ctx in request options."""
    from unittest.mock import MagicMock, patch
    from src.agent.llm import OllamaClient
    from src.config import settings

    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:1.7b")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "test response"}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        res = client.generate("test prompt")
        assert res == "test response"
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert "options" in payload
        assert payload["options"].get("num_ctx") == settings.ollama_num_ctx
