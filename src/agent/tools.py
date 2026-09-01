from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from src.config import settings
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.models import RerankedResult
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract interface for agent tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass


class HybridSearchTool(BaseTool):
    """Tool to perform hybrid dense + sparse retrieval with RRF."""

    name = "hybrid_search"
    description = (
        "Search the local technical documentation using hybrid dense vector search "
        "and sparse BM25 keyword search with Reciprocal Rank Fusion."
    )

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever()

    def execute(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        target_k = top_k or settings.rerank_candidates_count
        logger.info(f"[Tool: hybrid_search] Query: '{query}', Candidate Top-K: {target_k}")
        return self.retriever.hybrid_search(query=query, top_k=target_k)


class RerankTool(BaseTool):
    """Tool to re-rank candidate documents using Cross-Encoder attention scoring."""

    name = "rerank"
    description = (
        "Re-score and re-rank candidate chunks against the user query using a "
        "cross-encoder neural network model to select the most relevant context."
    )

    def __init__(self, reranker: Optional[CrossEncoderReranker] = None):
        self.reranker = reranker or CrossEncoderReranker()

    def execute(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        target_k = top_k or settings.rerank_top_k
        logger.info(f"[Tool: rerank] Scoring {len(candidates)} candidates -> Final Top-K: {target_k}")
        return self.reranker.rerank(query=query, candidates=candidates, top_k=target_k)


class ToolRegistry:
    """Registry managing available agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]
