"""Tests for Phase 2: Conversational Memory and Contextual Query Reformulation."""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.agent.agent import LocalQwenAgent
from src.agent.llm import MockOllamaClient
from src.agent.tools import HybridSearchTool, RerankTool
from src.api.routes import router
from src.api.schemas import QueryRequest, QueryResponse
from src.config import settings
from src.reranking.models import RerankedResult
from src.retrieval.models import SearchResult
from src.ui.api_client import RAGApiClient


class MockRetrieverForHistory:
    """Retriever tracking all queries submitted during agent execution."""

    def __init__(self):
        self.recorded_queries: List[str] = []

    def hybrid_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        self.recorded_queries.append(query)
        return [
            SearchResult(
                chunk_id="chunk_k8s_pod_1",
                text="A Kubernetes Pod is the smallest deployable compute unit in Kubernetes.",
                source="kubernetes-pods.html",
                metadata={"filename": "kubernetes-pods.html", "section": "Pod Overview"},
                score=0.035,
                dense_rank=1,
                sparse_rank=1,
                rrf_score=0.035,
            ),
            SearchResult(
                chunk_id="chunk_k8s_lifecycle_1",
                text="The lifecycle of a Pod includes Pending, Running, Succeeded, Failed, and Unknown phases.",
                source="kubernetes-pod-lifecycle.html",
                metadata={"filename": "kubernetes-pod-lifecycle.html", "section": "Pod Lifecycle"},
                score=0.032,
                dense_rank=2,
                sparse_rank=2,
                rrf_score=0.032,
            ),
        ]


class MockRerankerForHistory:
    """Mock reranker preserving candidate chunks."""

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 5) -> List[RerankedResult]:
        return [
            RerankedResult(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                metadata=c.metadata,
                score=c.score,
                dense_rank=c.dense_rank,
                sparse_rank=c.sparse_rank,
                rrf_score=c.rrf_score,
                rerank_score=0.85,
                rerank_rank=i + 1,
            )
            for i, c in enumerate(candidates[:top_k])
        ]


@pytest.fixture
def history_agent_tools():
    retriever = MockRetrieverForHistory()
    search_tool = HybridSearchTool(retriever=retriever)
    rerank_tool = RerankTool(reranker=MockRerankerForHistory())
    return retriever, search_tool, rerank_tool


# =========================================================================
# 1. Unit Tests for Chat History Bounding
# =========================================================================

def test_bound_chat_history_limits_turns(history_agent_tools):
    _, search_tool, rerank_tool = history_agent_tools
    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(),
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    # 5 turns = 10 messages
    long_history = [
        {"role": "user", "content": f"Question {i}"}
        if i % 2 == 0
        else {"role": "assistant", "content": f"Answer {i}"}
        for i in range(10)
    ]

    bounded = agent._bound_chat_history(long_history, max_turns=2)
    # At most 2 turns = 4 messages (the most recent 4)
    assert len(bounded) == 4
    assert bounded[0]["content"] == "Question 6"
    assert bounded[1]["content"] == "Answer 7"
    assert bounded[2]["content"] == "Question 8"
    assert bounded[3]["content"] == "Answer 9"


def test_bound_chat_history_truncates_long_assistant_messages(history_agent_tools):
    _, search_tool, rerank_tool = history_agent_tools
    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(),
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    long_answer = "x" * 500
    history = [
        {"role": "user", "content": "What is Docker?"},
        {"role": "assistant", "content": long_answer},
    ]

    bounded = agent._bound_chat_history(history, max_turns=1, max_chars_per_turn=200)
    assert len(bounded) == 2
    assert bounded[0]["content"] == "What is Docker?"
    assert len(bounded[1]["content"]) == 203  # 200 chars + "..."
    assert bounded[1]["content"].endswith("...")


def test_bound_chat_history_handles_question_answer_dict_format(history_agent_tools):
    _, search_tool, rerank_tool = history_agent_tools
    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(),
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    history = [
        {"question": "What is Docker?", "answer": "Docker is a container platform."},
    ]

    bounded = agent._bound_chat_history(history, max_turns=1)
    assert len(bounded) == 2
    assert bounded[0] == {"role": "user", "content": "What is Docker?"}
    assert bounded[1] == {"role": "assistant", "content": "Docker is a container platform."}


# =========================================================================
# 2. Unit Tests for Conversational Reformulation Across Domains
# =========================================================================

def test_followup_question_requiring_previous_context(history_agent_tools):
    """Verify follow-up question is rewritten into a self-contained search query."""
    _, search_tool, rerank_tool = history_agent_tools

    def llm_reformulator(prompt: str) -> str:
        if "Recent Conversation History:" in prompt and "its lifecycle" in prompt:
            return "What is the lifecycle of a Kubernetes Pod?"
        return "Grounded factual answer about pod lifecycle."

    mock_llm = MockOllamaClient(response_generator=llm_reformulator)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    history = [
        {"role": "user", "content": "What is a Kubernetes Pod?"},
        {"role": "assistant", "content": "A Pod is the smallest execution unit in Kubernetes."},
    ]

    reformulated = agent.reformulate_conversational_query("What about its lifecycle?", history)
    assert "Kubernetes Pod" in reformulated
    assert "lifecycle" in reformulated.lower()


def test_standalone_question_preserves_query(history_agent_tools):
    """Verify standalone question with empty history is unchanged."""
    _, search_tool, rerank_tool = history_agent_tools
    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(),
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    # Empty history
    reformulated = agent.reformulate_conversational_query("What is Kubernetes?", [])
    assert reformulated == "What is Kubernetes?"

    # route_and_rewrite without history preserves query
    needed, query = agent.route_and_rewrite("What is Kubernetes?", chat_history=[])
    assert needed is True
    assert query == "What is Kubernetes?"


def test_standalone_question_in_turn2_preserves_topic_switch(history_agent_tools):
    """Verify topic switch preserves the new standalone question."""
    _, search_tool, rerank_tool = history_agent_tools

    def llm_reformulator(prompt: str) -> str:
        if "Recent Conversation History:" in prompt and "Docker Swarm" in prompt:
            # The prompt instructs returning standalone queries as-is
            return "What is Docker Swarm?"
        return "Grounded factual answer."

    mock_llm = MockOllamaClient(response_generator=llm_reformulator)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    history = [
        {"role": "user", "content": "What is a Kubernetes Pod?"},
        {"role": "assistant", "content": "A Pod is the smallest execution unit in Kubernetes."},
    ]

    reformulated = agent.reformulate_conversational_query("What is Docker Swarm?", history)
    assert reformulated == "What is Docker Swarm?"


def test_followup_with_no_useful_previous_context(history_agent_tools):
    """Verify generic/greeting history does not corrupt a standalone question."""
    _, search_tool, rerank_tool = history_agent_tools

    def llm_reformulator(prompt: str) -> str:
        return "What is Kubernetes?"

    mock_llm = MockOllamaClient(response_generator=llm_reformulator)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
    ]

    reformulated = agent.reformulate_conversational_query("What is Kubernetes?", history)
    assert reformulated == "What is Kubernetes?"


def test_docker_domain_followup_reformulation(history_agent_tools):
    """Verify conversational reformulation works on Docker documentation without hardcoding."""
    _, search_tool, rerank_tool = history_agent_tools

    def llm_reformulator(prompt: str) -> str:
        if "bridge network" in prompt and "connect to it" in prompt:
            return "How to connect containers to a user-defined Docker bridge network"
        return "Docker bridge connect documentation."

    mock_llm = MockOllamaClient(response_generator=llm_reformulator)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    history = [
        {"role": "user", "content": "How do I create a custom bridge network in Docker?"},
        {"role": "assistant", "content": "Use `docker network create --driver bridge my-net`."},
    ]

    reformulated = agent.reformulate_conversational_query("How can containers connect to it?", history)
    assert "Docker bridge network" in reformulated
    assert "connect" in reformulated.lower()


def test_arbitrary_document_followup_reformulation(history_agent_tools):
    """Verify conversational reformulation works for arbitrary uploaded documentation."""
    _, search_tool, rerank_tool = history_agent_tools

    def llm_reformulator(prompt: str) -> str:
        if "PodDisruptionBudget" in prompt and "maxUnavailable" in prompt:
            return "PodDisruptionBudget maxUnavailable configuration specification"
        return "PDB documentation."

    mock_llm = MockOllamaClient(response_generator=llm_reformulator)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    history = [
        {"role": "user", "content": "What is a PodDisruptionBudget?"},
        {"role": "assistant", "content": "A PDB specifies the minimum available or maximum unavailable pods."},
    ]

    reformulated = agent.reformulate_conversational_query("How do I configure maxUnavailable?", history)
    assert "PodDisruptionBudget" in reformulated
    assert "maxUnavailable" in reformulated


# =========================================================================
# 3. Integration Test: Turn 1 -> Turn 2 Demonstrating Retrieval Context
# =========================================================================

def test_turn1_turn2_integration_retrieval_query(history_agent_tools):
    """Demonstrate Turn 1 and Turn 2 where Turn 2 retrieves using context from Turn 1.
    
    Turn 1: "What is a Kubernetes Pod?"
    Turn 2: "What about its lifecycle?"
    Expected: The query passed to retrieval contains enough context to identify "Kubernetes Pod".
    """
    retriever, search_tool, rerank_tool = history_agent_tools

    def mock_llm_behavior(prompt: str) -> str:
        if "Recent Conversation History:" in prompt and "its lifecycle" in prompt:
            return "What is the lifecycle of a Kubernetes Pod?"
        if "User Question:" in prompt or "Instructions:" in prompt:
            return "A Kubernetes Pod passes through Pending, Running, Succeeded, and Failed phases."
        return "General answer."

    mock_llm = MockOllamaClient(response_generator=mock_llm_behavior)
    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    # --- Turn 1: Standalone Question ---
    response_turn1 = agent.run("What is a Kubernetes Pod?")
    assert response_turn1.query == "What is a Kubernetes Pod?"
    assert len(response_turn1.sources) > 0
    assert "What is a Kubernetes Pod?" in retriever.recorded_queries

    # --- Turn 2: Follow-up Question with Conversation History ---
    chat_history = [
        {"role": "user", "content": response_turn1.query},
        {"role": "assistant", "content": response_turn1.answer},
    ]

    response_turn2 = agent.run("What about its lifecycle?", chat_history=chat_history)

    # 1. Preserves original query in response
    assert response_turn2.query == "What about its lifecycle?"

    # 2. Rewritten query contains "Kubernetes Pod" and "lifecycle"
    assert len(response_turn2.rewritten_queries) > 0
    rewritten = response_turn2.rewritten_queries[0]
    assert "Kubernetes Pod" in rewritten
    assert "lifecycle" in rewritten.lower()

    # 3. The query passed to retrieval was the reformulated query
    retrieval_queries = retriever.recorded_queries
    assert any("Kubernetes Pod" in q and "lifecycle" in q.lower() for q in retrieval_queries)

    # 4. Response contains grounded sources and answer
    assert len(response_turn2.sources) > 0
    assert "Pending" in response_turn2.answer
    assert response_turn2.metadata["original_query"] == "What about its lifecycle?"
    assert "Kubernetes Pod" in response_turn2.metadata["retrieval_query"]


# =========================================================================
# 4. API & UI Client Tests for Conversational Memory
# =========================================================================

def test_query_request_schema_normalization():
    """Verify QueryRequest normalizes both role/content and question/answer formats."""
    # Standard format
    req1 = QueryRequest(
        question="What about its lifecycle?",
        chat_history=[{"role": "user", "content": "What is a Pod?"}],
    )
    assert len(req1.chat_history) == 1
    assert req1.chat_history[0].role == "user"

    # Streamlit question/answer format
    req2 = QueryRequest(
        question="What about its lifecycle?",
        chat_history=[{"question": "What is a Pod?", "answer": "A Pod is..."}],
    )
    assert len(req2.chat_history) == 2
    assert req2.chat_history[0].role == "user"
    assert req2.chat_history[0].content == "What is a Pod?"
    assert req2.chat_history[1].role == "assistant"
    assert req2.chat_history[1].content == "A Pod is..."


def test_ui_api_client_sends_chat_history():
    """Verify RAGApiClient sends chat_history in payload."""
    client = RAGApiClient(base_url="http://testserver")

    history = [
        {"role": "user", "content": "What is a Pod?"},
        {"role": "assistant", "content": "A Pod is a Kubernetes unit."},
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "question": "What about its lifecycle?",
        "answer": "Pod lifecycle phases...",
        "sources": [],
        "orchestration": {
            "retrieval_needed": True,
            "hops_executed": 1,
            "rewritten_queries": ["What is the lifecycle of a Kubernetes Pod?"],
        },
        "performance": {"total_latency_ms": 150.0},
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = client.query_rag("What about its lifecycle?", chat_history=history)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["question"] == "What about its lifecycle?"
        assert payload["chat_history"] == history
        assert "What is the lifecycle of a Kubernetes Pod?" in result["orchestration"]["rewritten_queries"]
