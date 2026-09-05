import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.agent.llm import OllamaConnectionError, OllamaError
from src.api.dependencies import get_rag_service
from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.api.service import RAGService
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and component readiness",
    description="Check API liveness and inspect readiness of storage files and Ollama service.",
)
async def health_check(
    service: RAGService = Depends(get_rag_service),
) -> HealthResponse:
    """Return API liveness and lightweight component readiness status."""
    readiness = await service.check_readiness()

    models = {
        "ollama_model": settings.ollama_model,
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
    }

    return HealthResponse(
        status="ok",
        version="0.1.0",
        readiness=readiness,
        models=models,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute Agentic RAG Query",
    description="Ask a technical question to the offline-first Hybrid Agentic RAG system.",
    responses={
        503: {
            "description": "Local LLM service (Ollama) is unreachable or timed out.",
            "content": {"application/json": {"example": {"detail": "Ollama service unavailable."}}},
        },
        422: {
            "description": "Validation error (e.g. empty or whitespace-only question).",
        },
    },
)
async def query_rag(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Process a user query through the hybrid agent with CRAG orchestration."""
    try:
        response = await service.query(request.question)
        return response
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error during /query: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local LLM service (Ollama) is unreachable. Please verify Ollama is running at {settings.ollama_base_url}.",
        )
    except OllamaError as e:
        logger.error(f"Ollama error during /query: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local LLM service encountered an error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error during /query execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request.",
        )


@router.post(
    "/query/stream",
    summary="Execute Agentic RAG Query with Streaming Output",
    description="Stream thought events and synthesized token chunks via Server-Sent Events (SSE).",
)
async def query_rag_stream(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
):
    """Process a user query and stream tokens and status events via SSE."""
    from fastapi.responses import StreamingResponse

    try:
        return StreamingResponse(
            service.query_stream(request.question),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(f"Unexpected error during /query/stream execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the streaming request.",
        )
