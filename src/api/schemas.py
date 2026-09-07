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


class ChatMessage(BaseModel):
    """A single message in the conversational history."""

    role: str = Field(description="Role of the sender: 'user' or 'assistant'.")
    content: str = Field(description="Text content of the message.")


class QueryRequest(BaseModel):
    """User query payload."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The technical question to answer.",
        examples=["How does Docker bridge networking work?"],
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Recent conversation turns (at most 1-2 turns).",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters such as doc_type, filename, or section.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Optional flag to stream LLM response tokens.",
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question cannot be empty or whitespace only.")
        return stripped

    @field_validator("chat_history", mode="before")
    @classmethod
    def normalize_chat_history(cls, v: Any) -> Any:
        if not v:
            return []
        if isinstance(v, list):
            normalized = []
            for item in v:
                if isinstance(item, dict):
                    if "role" in item and "content" in item:
                        normalized.append(item)
                    elif "question" in item or "answer" in item:
                        if item.get("question"):
                            normalized.append({"role": "user", "content": str(item["question"])})
                        if item.get("answer"):
                            normalized.append({"role": "assistant", "content": str(item["answer"])})
                else:
                    normalized.append(item)
            return normalized
        return v


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


class DocumentUploadResponse(BaseModel):
    """Response returned upon document upload and indexing."""

    status: str = Field(description="Upload and indexing status ('success' or 'skipped').")
    filename: str = Field(description="Name of the uploaded document.")
    chunks_indexed: int = Field(description="Number of chunks extracted and indexed from this document.")
    total_chunks: int = Field(description="Total chunks in the corpus after indexing.")
    message: str = Field(description="Status description or outcome.")
