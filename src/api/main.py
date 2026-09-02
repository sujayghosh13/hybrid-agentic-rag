from contextlib import asynccontextmanager
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.dependencies import get_rag_service
from src.api.routes import router
from src.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Initializing Hybrid-Agentic RAG API backend...")
    yield
    logger.info("Shutting down Hybrid-Agentic RAG API backend...")
    # Clean up Qdrant client connection if service was initialized
    try:
        service = get_rag_service()
        if service._agent is not None:
            if hasattr(service._agent, "search_tool") and hasattr(service._agent.search_tool, "retriever"):
                service._agent.search_tool.retriever.close()
                logger.info("Closed hybrid retriever connections.")
    except Exception as e:
        logger.debug(f"Retriever teardown notice: {e}")


def create_app() -> FastAPI:
    """Application factory for the Hybrid-Agentic RAG API."""
    app = FastAPI(
        title="Hybrid-Agentic RAG API",
        version="0.1.0",
        description="Offline-first Technical Documentation QA Backend with Corrective RAG (CRAG).",
        lifespan=lifespan,
    )

    # Restrictive CORS configuration for local development / Phase 8 frontend
    allowed_origins = settings.cors_origins_list
    logger.info(f"Configuring CORS with allowed origins: {allowed_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
