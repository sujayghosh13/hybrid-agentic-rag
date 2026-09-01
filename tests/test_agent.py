import pytest

from src.agent.agent import LocalQwenAgent
from src.agent.llm import MockOllamaClient, OllamaConnectionError
from src.agent.models import AgentResponse, ToolCall
from src.agent.tools import BaseTool, HybridSearchTool, RerankTool, ToolRegistry
from src.reranking.models import RerankedResult
from src.retrieval.models import SearchResult


class MockHybridRetriever:
    """Mock HybridRetriever returning fixed candidate search results."""

    def hybrid_search(self, query: str, top_k: int = 20):
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
    assert len(response.sources) == 2  # Retrieval succeeded before LLM error
