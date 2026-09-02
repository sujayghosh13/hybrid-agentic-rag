from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from src.agent.llm import OllamaConnectionError, OllamaError
from src.agent.models import AgentResponse
from src.api.dependencies import get_rag_service
from src.api.main import app
from src.api.schemas import ReadinessStatus
from src.api.service import RAGService
from src.correction.models import EvidenceGrade


@pytest.fixture
def mock_agent():
    """Mock LocalQwenAgent producing predictable responses without Ollama."""
    agent = MagicMock()
    return agent


@pytest.fixture
def mock_service(mock_agent):
    """Mock RAGService wrapping mock_agent."""
    service = RAGService(agent=mock_agent)
    # Stub check_readiness to return true for unit tests
    async def mock_readiness():
        return ReadinessStatus(
            bm25_index_ready=True,
            qdrant_storage_ready=True,
            ollama_reachable=True,
        )
    service.check_readiness = mock_readiness
    return service


@pytest.fixture
def client(mock_service):
    """TestClient with dependency overrides to isolate the test environment."""
    app.dependency_overrides[get_rag_service] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# =====================================================================
# 1. Health Endpoint Tests
# =====================================================================


def test_health_check(client):
    """GET /health must return 200 with status ok and readiness metrics."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "readiness" in data
    assert data["readiness"]["bm25_index_ready"] is True
    assert data["readiness"]["qdrant_storage_ready"] is True
    assert data["readiness"]["ollama_reachable"] is True
    assert "models" in data
    assert "ollama_model" in data["models"]
    assert "embedding_model" in data["models"]
    assert "reranker_model" in data["models"]


# =====================================================================
# 2. Query Endpoint Tests (Successful Scenarios)
# =====================================================================


def test_query_success(client, mock_agent):
    """POST /query with valid question returns 200 and structured response."""
    mock_agent.run.return_value = AgentResponse(
        query="How does Docker bridge networking work?",
        answer="Docker bridge networking creates a virtual interface docker0 connecting containers.",
        sources=[
            {
                "chunk_id": "docker-bridge-network.html_chunk_9",
                "source": "data/raw/docker-bridge-network.html",
                "text": "User-defined bridges provide automatic DNS resolution...",
                "rerank_score": 0.892,
                "metadata": {"filename": "docker-bridge-network.html"},
            }
        ],
        hops_executed=1,
        retrieval_needed=True,
        final_evidence_grade=EvidenceGrade.GOOD,
        is_corrected=False,
        rewritten_queries=["Docker bridge network docker0"],
    )

    payload = {"question": "How does Docker bridge networking work?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "How does Docker bridge networking work?"
    assert "docker0" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "docker-bridge-network.html_chunk_9"
    assert data["sources"][0]["rerank_score"] == 0.892

    # Orchestration metadata
    assert data["orchestration"]["hops_executed"] == 1
    assert data["orchestration"]["final_evidence_grade"] == "GOOD"
    assert data["orchestration"]["is_corrected"] is False
    assert data["orchestration"]["rewritten_queries"] == ["Docker bridge network docker0"]

    # Performance metadata
    assert "total_latency_ms" in data["performance"]
    assert data["performance"]["total_latency_ms"] >= 0.0


def test_query_refusal_answer(client, mock_agent):
    """POST /query for ungrounded question returns 200 with standard refusal text."""
    refusal_text = "Based on the available local technical documentation, there is insufficient evidence to answer this question."
    mock_agent.run.return_value = AgentResponse(
        query="Tell me about AWS Lambda",
        answer=refusal_text,
        sources=[],
        hops_executed=1,
        retrieval_needed=True,
        final_evidence_grade=EvidenceGrade.BAD,
        is_corrected=False,
    )

    payload = {"question": "Tell me about AWS Lambda"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "insufficient evidence" in data["answer"]
    assert len(data["sources"]) == 0
    assert data["orchestration"]["final_evidence_grade"] == "BAD"


# =====================================================================
# 3. Input Validation Tests (422 Unprocessable Entity)
# =====================================================================


def test_query_validation_empty_string(client):
    """POST /query with empty string must return 422."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_validation_whitespace_only(client):
    """POST /query with whitespace-only must return 422."""
    response = client.post("/query", json={"question": "    "})
    assert response.status_code == 422


def test_query_validation_missing_field(client):
    """POST /query with missing question field must return 422."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_validation_invalid_type(client):
    """POST /query with invalid non-string question must return 422."""
    response = client.post("/query", json={"question": 12345})
    assert response.status_code == 422


# =====================================================================
# 4. Error Handling Tests (503 Service Unavailable & 500 Internal Error)
# =====================================================================


def test_query_ollama_connection_error(client, mock_agent):
    """POST /query when Ollama is unreachable must return 503."""
    mock_agent.run.side_effect = OllamaConnectionError("Connection timed out at localhost:11434")

    response = client.post("/query", json={"question": "Valid technical question"})
    assert response.status_code == 503
    data = response.json()
    assert "Local LLM service (Ollama) is unreachable" in data["detail"]


def test_query_ollama_generic_error(client, mock_agent):
    """POST /query when Ollama returns an error must return 503."""
    mock_agent.run.side_effect = OllamaError("Model runner failed")

    response = client.post("/query", json={"question": "Valid technical question"})
    assert response.status_code == 503
    data = response.json()
    assert "Local LLM service encountered an error" in data["detail"]


def test_query_unexpected_exception(client, mock_agent):
    """POST /query encountering an unexpected exception must return 500."""
    mock_agent.run.side_effect = RuntimeError("Disk full")

    response = client.post("/query", json={"question": "Valid technical question"})
    assert response.status_code == 500
    data = response.json()
    assert "An unexpected error occurred" in data["detail"]


# =====================================================================
# 5. CORS Header Tests
# =====================================================================


def test_cors_headers_allowed_origin(client):
    """Requests from configured origin (http://localhost:8501) receive CORS headers."""
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"
