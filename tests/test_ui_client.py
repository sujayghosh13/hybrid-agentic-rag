from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.ui.api_client import (
    APIConnectionError,
    APIServerError,
    APITimeoutError,
    APIValidationError,
    BackendUnavailableError,
    RAGApiClient,
)


@pytest.fixture
def client():
    """Create test client pointing to dummy base url."""
    return RAGApiClient(base_url="http://127.0.0.1:8000", timeout=5.0)


# =====================================================================
# 1. Health Check Tests
# =====================================================================


def test_health_check_success(client):
    """GET /health 200 OK returns parsed JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "version": "0.1.0",
        "readiness": {
            "bm25_index_ready": True,
            "qdrant_storage_ready": True,
            "ollama_reachable": True,
        },
        "models": {
            "ollama_model": "qwen3:4b",
            "embedding_model": "BGE-small",
            "reranker_model": "MiniLM",
        },
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        data = client.check_health()
        assert data["status"] == "ok"
        assert data["readiness"]["ollama_reachable"] is True
        assert data["models"]["ollama_model"] == "qwen3:4b"


def test_health_check_connection_error(client):
    """GET /health raises APIConnectionError when backend is offline."""
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(APIConnectionError) as exc_info:
            client.check_health()
        assert "Could not connect to FastAPI backend" in str(exc_info.value)


def test_health_check_timeout(client):
    """GET /health raises APITimeoutError on timeout."""
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(APITimeoutError) as exc_info:
            client.check_health()
        assert "timed out" in str(exc_info.value).lower()


# =====================================================================
# 2. Query Tests
# =====================================================================


def test_query_rag_success(client):
    """POST /query 200 OK returns structured response dictionary."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "question": "How does Docker bridge networking work?",
        "answer": "Bridge networking creates a software bridge docker0...",
        "sources": [
            {
                "chunk_id": "docker-bridge-network.html_chunk_9",
                "source": "data/raw/docker-bridge-network.html",
                "text": "User-defined bridges provide automatic DNS resolution...",
                "rerank_score": 0.892,
                "metadata": {"filename": "docker-bridge-network.html"},
            }
        ],
        "orchestration": {
            "retrieval_needed": True,
            "hops_executed": 1,
            "final_evidence_grade": "GOOD",
            "is_corrected": False,
            "rewritten_queries": ["Docker bridge network docker0"],
        },
        "performance": {"total_latency_ms": 142.5},
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        result = client.query_rag("How does Docker bridge networking work?")
        assert result["question"] == "How does Docker bridge networking work?"
        assert "docker0" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["rerank_score"] == 0.892
        assert result["orchestration"]["final_evidence_grade"] == "GOOD"
        assert result["performance"]["total_latency_ms"] == 142.5


def test_query_rag_422_validation_error(client):
    """POST /query 422 raises APIValidationError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": "question cannot be empty or whitespace only."}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(APIValidationError) as exc_info:
            client.query_rag("   ")
        assert "Validation error" in str(exc_info.value)


def test_query_rag_503_backend_unavailable(client):
    """POST /query 503 raises BackendUnavailableError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"detail": "Local LLM service (Ollama) is unreachable."}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(BackendUnavailableError) as exc_info:
            client.query_rag("How does Docker work?")
        assert "Backend LLM service unavailable" in str(exc_info.value)


def test_query_rag_500_server_error(client):
    """POST /query 500 raises APIServerError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"detail": "Internal server error"}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(APIServerError) as exc_info:
            client.query_rag("Valid question")
        assert "unexpected internal error" in str(exc_info.value)


def test_query_rag_connection_error(client):
    """POST /query raises APIConnectionError when backend is offline."""
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(APIConnectionError) as exc_info:
            client.query_rag("Valid question")
        assert "Could not connect to FastAPI backend" in str(exc_info.value)


def test_query_rag_timeout(client):
    """POST /query raises APITimeoutError when query times out."""
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(APITimeoutError) as exc_info:
            client.query_rag("Complex question")
        assert "timed out" in str(exc_info.value).lower()
