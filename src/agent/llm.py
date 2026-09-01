from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx
from src.config import settings

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base exception for Ollama errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server is unreachable or timed out."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an error response."""
    pass


class BaseLLMClient(ABC):
    """Abstract interface for LLM client."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        pass


class OllamaClient(BaseLLMClient):
    """Client for local Ollama server running Qwen3."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.agent_temperature,
            },
        }
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}: {response.text}")
                data = response.json()
                return data.get("response", "").strip()
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at '{self.base_url}'. Is Ollama running? Error: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OllamaError):
                raise
            raise OllamaResponseError(f"Error calling Ollama: {e}") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.agent_temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}: {response.text}")
                data = response.json()
                msg = data.get("message", {})
                return msg.get("content", "").strip()
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at '{self.base_url}'. Is Ollama running? Error: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OllamaError):
                raise
            raise OllamaResponseError(f"Error calling Ollama chat API: {e}") from e


class MockOllamaClient(BaseLLMClient):
    """Mock LLM client for deterministic unit testing."""

    def __init__(
        self,
        response_generator: Optional[Callable[[str], str]] = None,
        default_response: str = "Mock answer generated from context.",
    ):
        self.response_generator = response_generator
        self.default_response = default_response
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self.call_history.append({"prompt": prompt, "system": system, "temperature": temperature})
        if self.response_generator:
            return self.response_generator(prompt)
        return self.default_response

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        self.call_history.append({"messages": messages, "temperature": temperature})
        last_content = messages[-1]["content"] if messages else ""
        if self.response_generator:
            return self.response_generator(last_content)
        return self.default_response
