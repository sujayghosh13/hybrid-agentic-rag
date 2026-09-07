import time
from unittest.mock import MagicMock, patch
import pytest

from src.agent.agent import LocalQwenAgent
from src.agent.llm import MockOllamaClient, OllamaClient
from src.agent.tools import HybridSearchTool, RerankTool
from src.config import settings
from src.reranking.models import RerankedResult
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import SearchResult
from src.retrieval.sparse import BM25Retriever


class SlowDenseRetriever:
    def __init__(self, delay=0.05):
        self.delay = delay
        self.call_count = 0

    def search(self, query: str, top_k: int = 10, filters=None):
        self.call_count += 1
        time.sleep(self.delay)
        return [
            SearchResult(
                chunk_id="chunk_dense_1",
                text="Dense result text",
                source="doc1.html",
                metadata={},
                score=0.9,
                dense_rank=1,
            )
        ]

    def close(self):
        pass


class SlowSparseRetriever:
    def __init__(self, delay=0.05):
        self.delay = delay
        self.call_count = 0

    def search(self, query: str, top_k: int = 10, filters=None):
        self.call_count += 1
        time.sleep(self.delay)
        return [
            SearchResult(
                chunk_id="chunk_sparse_1",
                text="Sparse result text",
                source="doc2.html",
                metadata={},
                score=5.0,
                sparse_rank=1,
            )
        ]


def test_parallel_hybrid_search_speedup():
    """Verify that hybrid search runs dense and sparse retrieval concurrently."""
    delay = 0.06
    slow_dense = SlowDenseRetriever(delay=delay)
    slow_sparse = SlowSparseRetriever(delay=delay)

    hybrid = HybridRetriever(dense_retriever=slow_dense, sparse_retriever=slow_sparse)
    try:
        t0 = time.perf_counter()
        results = hybrid.hybrid_search("docker bridge network", top_k=2)
        elapsed = time.perf_counter() - t0

        # If sequential, elapsed >= 2 * delay (0.12s)
        # In parallel, elapsed should be significantly less than 2 * delay
        assert elapsed < (delay * 1.8), f"Expected parallel execution < {delay * 1.8}s, got {elapsed:.3f}s"
        assert len(results) > 0
        assert slow_dense.call_count == 1
        assert slow_sparse.call_count == 1
    finally:
        hybrid.close()


def test_dense_retriever_query_embedding_cache():
    """Verify that query embeddings are cached in memory across repeated queries."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.1] * 16

    mock_client = MagicMock()
    mock_client.search.return_value = []

    retriever = DenseRetriever(
        collection_name="test_col",
        client=mock_client,
        embedder=mock_embedder,
    )

    # First search: cache miss
    retriever.search("How to configure bridge network?", top_k=5)
    assert mock_embedder.encode.call_count == 2  # 1 for test dimension probe, 1 for search

    # Second search with identical query: cache hit, no additional encode call
    retriever.search("How to configure bridge network?", top_k=5)
    assert mock_embedder.encode.call_count == 2


def test_ollama_client_persistent_connection_pool():
    """Verify OllamaClient reuses a persistent httpx.Client across multiple calls."""
    client = OllamaClient(base_url="http://localhost:11434")
    try:
        http1 = client.http_client
        http2 = client.http_client
        assert http1 is http2
        assert not http1.is_closed
    finally:
        client.close()
        assert http1.is_closed


def test_agent_run_stream_yields_realtime_tokens():
    """Verify LocalQwenAgent.run_stream yields metadata, tokens, and done events."""
    mock_llm = MockOllamaClient(default_response="Docker bridge facilitates container communication.")
    mock_search = MagicMock()
    mock_search.execute.return_value = []
    mock_rerank = MagicMock()
    mock_rerank.execute.return_value = []

    agent = LocalQwenAgent(
        llm_client=mock_llm,
        hybrid_search_tool=mock_search,
        rerank_tool=mock_rerank,
    )

    events = list(agent.run_stream("hello"))
    event_types = [e["type"] for e in events]
    assert "metadata" in event_types
    assert "token" in event_types
    assert "done" in event_types
