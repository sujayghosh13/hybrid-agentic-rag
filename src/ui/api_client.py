import logging
from typing import Any, Dict, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class RAGClientError(Exception):
    """Base exception for all Streamlit API client errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class APIConnectionError(RAGClientError):
    """Raised when the FastAPI backend is completely offline or unreachable."""
    pass


class APITimeoutError(RAGClientError):
    """Raised when the backend request exceeds the timeout threshold."""
    pass


class APIValidationError(RAGClientError):
    """Raised on HTTP 422 input validation errors."""
    pass


class BackendUnavailableError(RAGClientError):
    """Raised on HTTP 503 when local Ollama or LLM service is offline."""
    pass


class APIServerError(RAGClientError):
    """Raised on HTTP 500 internal server errors."""
    pass


class RAGApiClient:
    """Pure HTTP client communicating with the Phase 7 FastAPI backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 180.0,
    ):
        self.base_url = (base_url or settings.fastapi_base_url).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Query GET /health to verify API liveness and component readiness.

        Returns:
            Dict containing 'status', 'version', 'readiness', and 'models'.
        """
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.get(url)
                if res.status_code == 200:
                    return res.json()
                raise APIServerError(
                    f"Health check failed with unexpected status code {res.status_code}",
                    details=res.text,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise APIConnectionError(
                f"Could not connect to FastAPI backend at '{self.base_url}'.",
                details=str(e),
            ) from e
        except httpx.TimeoutException as e:
            raise APITimeoutError(
                f"Health check timed out after 10.0 seconds.",
                details=str(e),
            ) from e
        except Exception as e:
            if isinstance(e, RAGClientError):
                raise
            raise RAGClientError(f"Unexpected error checking API health: {e}") from e

    def query_rag(self, question: str) -> Dict[str, Any]:
        """Query POST /query to execute the agentic RAG pipeline.

        Args:
            question: Technical question string.

        Returns:
            Dict containing 'question', 'answer', 'sources', 'orchestration', 'performance'.
        """
        url = f"{self.base_url}/query"
        payload = {"question": question}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)

                if res.status_code == 200:
                    return res.json()

                if res.status_code == 422:
                    error_detail = "Invalid input query."
                    try:
                        data = res.json()
                        error_detail = data.get("detail", error_detail)
                    except Exception:
                        pass
                    raise APIValidationError(
                        "Validation error: The question submitted was empty or invalid.",
                        details=str(error_detail),
                    )

                if res.status_code == 503:
                    error_detail = "Local LLM service (Ollama) is unavailable."
                    try:
                        data = res.json()
                        error_detail = data.get("detail", error_detail)
                    except Exception:
                        pass
                    raise BackendUnavailableError(
                        "Backend LLM service unavailable.",
                        details=str(error_detail),
                    )

                if res.status_code == 500:
                    error_detail = "Internal server error."
                    try:
                        data = res.json()
                        error_detail = data.get("detail", error_detail)
                    except Exception:
                        pass
                    raise APIServerError(
                        "Backend encountered an unexpected internal error.",
                        details=str(error_detail),
                    )

                raise APIServerError(
                    f"Unexpected HTTP status {res.status_code} from backend.",
                    details=res.text,
                )

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise APIConnectionError(
                f"Could not connect to FastAPI backend at '{self.base_url}'.",
                details=str(e),
            ) from e
        except httpx.TimeoutException as e:
            raise APITimeoutError(
                f"Query request timed out after {self.timeout:.0f} seconds. "
                "Local inference may be under heavy load.",
                details=str(e),
            ) from e
        except Exception as e:
            if isinstance(e, RAGClientError):
                raise
            raise RAGClientError(f"Unexpected error querying RAG API: {e}") from e
