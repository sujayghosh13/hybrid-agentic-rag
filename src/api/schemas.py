from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ReadinessStatus(BaseModel):
    """Readiness indicators for underlying backend components."""

    bm25_index_ready: bool
    qdrant_storage_ready: bool
    ollama_reachable: bool


class HealthResponse(BaseModel):
    """Liveness and lightweight component readiness status."""

    status: str = "ok"
    version: str = "0.1.0"
    readiness: ReadinessStatus
    models: Dict[str, str]


class QueryRequest(BaseModel):
    """User query payload."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The technical question to answer.",
        examples=["How does Docker bridge networking work?"],
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question cannot be empty or whitespace only.")
        return stripped


class SourceItem(BaseModel):
    """Individual retrieved document chunk attributed in the final answer."""

    chunk_id: str
    source: str
    text: str
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestrationMetadata(BaseModel):
    """Production-curated agent and CRAG decision metadata."""

    retrieval_needed: bool
    hops_executed: int
    final_evidence_grade: Optional[str] = None
    is_corrected: bool = False
    rewritten_queries: List[str] = Field(default_factory=list)


class PerformanceMetadata(BaseModel):
    """Measured end-to-end execution latency."""

    total_latency_ms: float


class QueryResponse(BaseModel):
    """Structured response returned by the RAG backend."""

    question: str
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    orchestration: OrchestrationMetadata
    performance: PerformanceMetadata
