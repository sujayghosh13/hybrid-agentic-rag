import logging
from typing import Optional

from src.api.service import RAGService

logger = logging.getLogger(__name__)

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Dependency provider returning the shared RAGService instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def set_rag_service(service: Optional[RAGService]) -> None:
    """Set or reset the shared RAGService instance (useful for testing)."""
    global _rag_service
    _rag_service = service
