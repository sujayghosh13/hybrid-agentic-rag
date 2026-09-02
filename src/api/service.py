import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from src.agent.agent import LocalQwenAgent
from src.agent.models import AgentResponse
from src.api.schemas import (
    OrchestrationMetadata,
    PerformanceMetadata,
    QueryResponse,
    ReadinessStatus,
    SourceItem,
)
from src.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service layer coordinating queries through LocalQwenAgent outside the event loop."""

    def __init__(self, agent: Optional[LocalQwenAgent] = None):
        self._agent = agent

    @property
    def agent(self) -> LocalQwenAgent:
        """Lazy-initialize LocalQwenAgent on first query request."""
        if self._agent is None:
            logger.info("Lazy-initializing LocalQwenAgent...")
            self._agent = LocalQwenAgent()
        return self._agent

    async def query(self, question: str) -> QueryResponse:
        """Execute a user query through the agent in a non-blocking threadpool."""
        clean_question = question.strip()
        t0 = time.perf_counter()

        # Run synchronous, CPU-intensive agent pipeline in threadpool
        response: AgentResponse = await asyncio.to_thread(self.agent.run, clean_question)

        t1 = time.perf_counter()
        total_latency_ms = (t1 - t0) * 1000.0

        # Map sources into clean API DTOs
        sources = [
            SourceItem(
                chunk_id=s.get("chunk_id", ""),
                source=s.get("source", ""),
                text=s.get("text", ""),
                rerank_score=s.get("rerank_score"),
                metadata=s.get("metadata", {}),
            )
            for s in response.sources
        ]

        # Extract final evidence grade
        grade_str = (
            response.final_evidence_grade.value
            if hasattr(response.final_evidence_grade, "value")
            else str(response.final_evidence_grade)
            if response.final_evidence_grade
            else None
        )

        orchestration = OrchestrationMetadata(
            retrieval_needed=response.retrieval_needed,
            hops_executed=response.hops_executed,
            final_evidence_grade=grade_str,
            is_corrected=response.is_corrected,
            rewritten_queries=response.rewritten_queries,
        )

        performance = PerformanceMetadata(total_latency_ms=round(total_latency_ms, 2))

        return QueryResponse(
            question=clean_question,
            answer=response.answer,
            sources=sources,
            orchestration=orchestration,
            performance=performance,
        )

    async def check_readiness(self) -> ReadinessStatus:
        """Lightweight readiness checks for local files and Ollama service."""
        # 1. Check BM25 index file
        bm25_ready = Path(settings.bm25_index_path).exists()

        # 2. Check Qdrant storage directory
        qdrant_ready = Path("data/processed/qdrant_storage").exists()

        # 3. Lightweight check for Ollama API (non-blocking, 1-second timeout)
        ollama_ready = False
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                res = await client.get(f"{settings.ollama_base_url}/api/tags")
                ollama_ready = res.status_code == 200
        except Exception:
            ollama_ready = False

        return ReadinessStatus(
            bm25_index_ready=bm25_ready,
            qdrant_storage_ready=qdrant_ready,
            ollama_reachable=ollama_ready,
        )
