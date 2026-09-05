from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
import json
from src.config import settings

logger = logging.getLogger(__name__)


def clean_llm_response(text: str) -> str:
    """Strip <think>...</think> reasoning tags and whitespace from LLM output."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def call_llm_generate(llm_client: Any, prompt: str, **kwargs) -> str:
    """Invoke llm.generate, gracefully falling back if a mock only supports basic args."""
    try:
        return llm_client.generate(prompt=prompt, **kwargs)
    except TypeError:
        fallback_kwargs = {k: v for k, v in kwargs.items() if k in ("system", "temperature")}
        return llm_client.generate(prompt=prompt, **fallback_kwargs)


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
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        pass


class OllamaClient(BaseLLMClient):
    """Client for local Ollama server running Qwen3 / Qwen2.5."""

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
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else settings.agent_temperature,
        }
        if getattr(settings, "ollama_num_ctx", None):
            options["num_ctx"] = settings.ollama_num_ctx
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop is not None:
            options["stop"] = stop

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}: {response.text}")
                data = response.json()
                raw_res = data.get("response", "")
                # If thinking model generated thought but response field is empty, fallback to thinking
                if not raw_res.strip() and data.get("thinking"):
                    raw_res = data.get("thinking", "")
                return clean_llm_response(raw_res)
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
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else settings.agent_temperature,
        }
        if getattr(settings, "ollama_num_ctx", None):
            options["num_ctx"] = settings.ollama_num_ctx
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop is not None:
            options["stop"] = stop

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}: {response.text}")
                data = response.json()
                msg = data.get("message", {})
                content = msg.get("content", "")
                if not content.strip() and msg.get("thinking"):
                    content = msg.get("thinking", "")
                return clean_llm_response(content)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at '{self.base_url}'. Is Ollama running? Error: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OllamaError):
                raise
            raise OllamaResponseError(f"Error calling Ollama chat API: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[str]:
        url = f"{self.base_url}/api/generate"
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else settings.agent_temperature,
        }
        if getattr(settings, "ollama_num_ctx", None):
            options["num_ctx"] = settings.ollama_num_ctx
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop is not None:
            options["stop"] = stop

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at '{self.base_url}'. Is Ollama running? Error: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OllamaError):
                raise
            raise OllamaResponseError(f"Error streaming from Ollama: {e}") from e


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
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        call_entry = {"prompt": prompt, "system": system, "temperature": temperature}
        if max_tokens is not None:
            call_entry["max_tokens"] = max_tokens
        if stop is not None:
            call_entry["stop"] = stop
        self.call_history.append(call_entry)
        if self.response_generator:
            return self.response_generator(prompt)
        return self.default_response

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[str]:
        full_res = self.generate(
            prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs
        )
        for token in full_res.split(" "):
            yield token + " "

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        call_entry = {"messages": messages, "temperature": temperature}
        if max_tokens is not None:
            call_entry["max_tokens"] = max_tokens
        if stop is not None:
            call_entry["stop"] = stop
        self.call_history.append(call_entry)
        last_content = messages[-1]["content"] if messages else ""
        if self.response_generator:
            return self.response_generator(last_content)
        return self.default_response
