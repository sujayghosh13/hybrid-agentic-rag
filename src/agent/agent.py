import logging
import time
from typing import Any, Dict, List, Optional

from src.agent.llm import BaseLLMClient, OllamaClient, OllamaConnectionError, OllamaError
from src.agent.models import AgentResponse, ToolCall
from src.agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_prompt,
)
from src.agent.tools import BaseTool, HybridSearchTool, RerankTool, ToolRegistry
from src.config import settings

logger = logging.getLogger(__name__)


class LocalQwenAgent:
    """Local offline-first Agent powered by Qwen3 via Ollama."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        hybrid_search_tool: Optional[HybridSearchTool] = None,
        rerank_tool: Optional[RerankTool] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        self.llm = llm_client or OllamaClient()
        self.search_tool = hybrid_search_tool or HybridSearchTool()
        self.rerank_tool = rerank_tool or RerankTool()

        self.registry = registry or ToolRegistry()
        self.registry.register(self.search_tool)
        self.registry.register(self.rerank_tool)

    def should_retrieve(self, query: str) -> bool:
        """Decide if local technical documentation retrieval is required for the query."""
        clean_q = query.strip().lower()

        # Conversational / greeting heuristic filter
        if clean_q in ("hello", "hi", "hey", "who are you?", "what can you do?", "thanks", "thank you"):
            return False

        # Technical domain terms fast-path
        tech_keywords = (
            "docker", "kubernetes", "k8s", "bridge", "network", "networking",
            "driver", "container", "pod", "pods", "service", "deployment",
            "daemon", "volume", "port", "ingress", "cluster", "workload",
            "image", "config", "ip", "subnet", "gateway", "how", "what", "why"
        )
        if any(kw in clean_q for kw in tech_keywords):
            return True

        try:
            routing_decision = self.llm.generate(
                prompt=f"User Query: {query}\nDecision:",
                system=ROUTER_SYSTEM_PROMPT,
                temperature=0.0,
            )
            decision = routing_decision.strip().upper()
            logger.info(f"[Agent Routing] Query: '{query}' -> Decision: '{decision}'")
            return "RETRIEVE" in decision
        except Exception as e:
            logger.warning(f"Routing classifier encountered error: {e}. Defaulting to RETRIEVE.")
            return True

    def run(self, query: str) -> AgentResponse:
        """Run the full agentic query workflow."""
        start_time = time.time()
        thought_process: List[str] = []
        tool_calls: List[ToolCall] = []

        if not query or not query.strip():
            return AgentResponse(
                query=query,
                answer="Please provide a valid question.",
                retrieval_needed=False,
                thought_process=["Empty query received."],
            )

        query = query.strip()
        thought_process.append(f"Received query: '{query}'")

        # Step 1: Decision on whether retrieval is needed
        retrieval_needed = self.should_retrieve(query)

        if not retrieval_needed:
            thought_process.append("Query classified as conversational/direct. Generating direct answer.")
            try:
                direct_answer = self.llm.generate(
                    prompt=query,
                    system="You are a helpful and polite technical AI assistant.",
                )
                return AgentResponse(
                    query=query,
                    answer=direct_answer,
                    retrieval_needed=False,
                    thought_process=thought_process,
                    metadata={"latency_sec": round(time.time() - start_time, 3)},
                )
            except OllamaConnectionError as e:
                return self._handle_ollama_connection_error(query, e, thought_process, tool_calls)
            except Exception as e:
                logger.error(f"Error during direct answer generation: {e}", exc_info=True)
                return AgentResponse(
                    query=query,
                    answer=f"An error occurred while generating the answer: {e}",
                    retrieval_needed=False,
                    thought_process=thought_process,
                    metadata={"error": str(e)},
                )

        # Step 2: Perform Hybrid Retrieval
        thought_process.append(f"Invoking Tool 'hybrid_search' for up to {settings.rerank_candidates_count} candidates.")
        try:
            candidates = self.search_tool.execute(query=query, top_k=settings.rerank_candidates_count)
            tool_calls.append(
                ToolCall(
                    tool_name="hybrid_search",
                    arguments={"query": query, "top_k": settings.rerank_candidates_count},
                    result=candidates,
                )
            )
            thought_process.append(f"Hybrid search returned {len(candidates)} candidate chunks.")
        except Exception as e:
            logger.error(f"Error in hybrid search tool: {e}", exc_info=True)
            return AgentResponse(
                query=query,
                answer=f"Error retrieving technical documentation: {e}",
                retrieval_needed=True,
                thought_process=thought_process,
                metadata={"error": str(e)},
            )

        if not candidates:
            thought_process.append("No matching candidates found in local knowledge base.")
            return AgentResponse(
                query=query,
                answer="Based on the available local documentation, no relevant information was found to answer this question.",
                retrieval_needed=True,
                tool_calls=tool_calls,
                thought_process=thought_process,
                metadata={"latency_sec": round(time.time() - start_time, 3)},
            )

        # Step 3: Perform Cross-Encoder Re-ranking
        thought_process.append(f"Invoking Tool 'rerank' to score and select top {settings.rerank_top_k} chunks.")
        try:
            reranked_chunks = self.rerank_tool.execute(
                query=query,
                candidates=candidates,
                top_k=settings.rerank_top_k,
            )
            tool_calls.append(
                ToolCall(
                    tool_name="rerank",
                    arguments={"query": query, "candidates_count": len(candidates), "top_k": settings.rerank_top_k},
                    result=reranked_chunks,
                )
            )
            thought_process.append(f"Re-ranking complete. Selected top {len(reranked_chunks)} context chunks.")
        except Exception as e:
            logger.error(f"Error in rerank tool: {e}", exc_info=True)
            return AgentResponse(
                query=query,
                answer=f"Error re-ranking technical documentation: {e}",
                retrieval_needed=True,
                tool_calls=tool_calls,
                thought_process=thought_process,
                metadata={"error": str(e)},
            )

        # Step 4: Synthesize Grounded Answer with LLM
        thought_process.append("Synthesizing final answer grounded on retrieved and re-ranked context.")
        synthesis_prompt = build_synthesis_prompt(query=query, context_chunks=reranked_chunks)

        try:
            answer = self.llm.generate(
                prompt=synthesis_prompt,
                system=SYNTHESIS_SYSTEM_PROMPT,
                temperature=settings.agent_temperature,
            )
        except OllamaConnectionError as e:
            return self._handle_ollama_connection_error(query, e, thought_process, tool_calls, reranked_chunks)
        except Exception as e:
            logger.error(f"Error during LLM answer synthesis: {e}", exc_info=True)
            return AgentResponse(
                query=query,
                answer=f"Error generating answer: {e}",
                retrieval_needed=True,
                sources=[chunk.to_dict() for chunk in reranked_chunks],
                tool_calls=tool_calls,
                thought_process=thought_process,
                metadata={"error": str(e)},
            )

        elapsed_time = round(time.time() - start_time, 3)
        thought_process.append(f"Answer synthesized successfully in {elapsed_time}s.")

        sources = [chunk.to_dict() for chunk in reranked_chunks]

        return AgentResponse(
            query=query,
            answer=answer,
            retrieval_needed=True,
            sources=sources,
            tool_calls=tool_calls,
            thought_process=thought_process,
            metadata={
                "latency_sec": elapsed_time,
                "model": getattr(self.llm, "model", "mock"),
                "chunks_retrieved": len(candidates),
                "chunks_used": len(reranked_chunks),
            },
        )

    def _handle_ollama_connection_error(
        self,
        query: str,
        error: OllamaConnectionError,
        thought_process: List[str],
        tool_calls: List[ToolCall],
        reranked_chunks: Optional[List[Any]] = None,
    ) -> AgentResponse:
        logger.error(f"Ollama server connection error: {error}")
        thought_process.append(f"Failed to connect to Ollama server: {error}")
        sources = [chunk.to_dict() for chunk in reranked_chunks] if reranked_chunks else []
        return AgentResponse(
            query=query,
            answer=(
                "Could not connect to the local Ollama LLM server. "
                "Please verify that Ollama is running (`ollama serve` or Ollama app) "
                f"and that model '{settings.ollama_model}' is available."
            ),
            retrieval_needed=True,
            sources=sources,
            tool_calls=tool_calls,
            thought_process=thought_process,
            metadata={"error": str(error), "ollama_status": "unreachable"},
        )
