import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.agent.llm import BaseLLMClient, OllamaClient, OllamaConnectionError, OllamaError
from src.agent.models import AgentResponse, HopTrace, ToolCall
from src.agent.prompts import (
    REWRITE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SUFFICIENCY_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_rewrite_prompt,
    build_sufficiency_prompt,
    build_synthesis_prompt,
)
from src.agent.tools import BaseTool, HybridSearchTool, RerankTool, ToolRegistry
from src.config import settings
from src.reranking.models import RerankedResult

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

    def rewrite_query(self, query: str, missing_aspect: Optional[str] = None) -> str:
        """Rewrite a user query into concise, high-signal retrieval keywords."""
        if not settings.query_rewriter_enabled:
            return query

        prompt = build_rewrite_prompt(query, missing_aspect=missing_aspect)
        try:
            rewritten = self.llm.generate(
                prompt=prompt,
                system=REWRITE_SYSTEM_PROMPT,
                temperature=0.0,
            )
            cleaned = rewritten.strip().strip('"').strip("'").split("\n")[0].strip()
            if cleaned and len(cleaned) > 2:
                logger.info(f"[Query Rewriter] '{query}' -> '{cleaned}' (missing: '{missing_aspect}')")
                return cleaned
        except Exception as e:
            logger.warning(f"Query rewriter encountered error: {e}. Falling back to original query.")

        return query

    def evaluate_sufficiency(
        self,
        query: str,
        context_chunks: List[RerankedResult],
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate if retrieved context is sufficient to answer the user query."""
        if not settings.sufficiency_check_enabled:
            return True, None

        if not context_chunks:
            return False, "No context chunks available"

        prompt = build_sufficiency_prompt(query, context_chunks)
        try:
            eval_response = self.llm.generate(
                prompt=prompt,
                system=SUFFICIENCY_SYSTEM_PROMPT,
                temperature=0.0,
            )
            decision_text = eval_response.strip()
            logger.info(f"[Sufficiency Check] Response: '{decision_text}'")

            if "SUFFICIENT" in decision_text.upper() and not decision_text.upper().startswith("INSUFFICIENT"):
                return True, None

            if "INSUFFICIENT" in decision_text.upper():
                parts = decision_text.split(":", 1)
                missing = parts[1].strip() if len(parts) > 1 else "additional technical context needed"
                return False, missing
        except Exception as e:
            logger.warning(f"Sufficiency check encountered error: {e}. Defaulting to SUFFICIENT.")

        return True, None

    def run(self, query: str) -> AgentResponse:
        """Run the full agentic query workflow with multi-hop retrieval and sufficiency check."""
        start_time = time.time()
        thought_process: List[str] = []
        tool_calls: List[ToolCall] = []
        hop_traces: List[HopTrace] = []
        rewritten_queries: List[str] = []

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

        # Step 2: Multi-Hop Retrieval Loop (Strictly bounded at max_hops <= 2)
        all_chunks_dict: Dict[str, RerankedResult] = {}
        missing_aspect: Optional[str] = None
        max_hops = min(settings.agent_max_hops, 2)
        executed_hops = 0

        for hop in range(1, max_hops + 1):
            executed_hops = hop
            thought_process.append(f"--- Starting Retrieval Hop {hop}/{max_hops} ---")

            # A. Query Rewriting / Optimization
            search_query = self.rewrite_query(query, missing_aspect=missing_aspect)
            rewritten_queries.append(search_query)

            # B. Duplicate Query Detection (Prevent infinite retrieval loops)
            if rewritten_queries.count(search_query) > 1:
                thought_process.append(f"Query '{search_query}' was already searched in an earlier hop. Breaking loop.")
                break

            # C. Hybrid Search Tool
            thought_process.append(f"Invoking 'hybrid_search' for query: '{search_query}' (Top {settings.rerank_candidates_count})")
            try:
                candidates = self.search_tool.execute(query=search_query, top_k=settings.rerank_candidates_count)
                tool_calls.append(
                    ToolCall(
                        tool_name="hybrid_search",
                        arguments={"query": search_query, "top_k": settings.rerank_candidates_count},
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
                thought_process.append(f"No candidates found for query: '{search_query}'.")
                break

            # D. Cross-Encoder Rerank Tool
            thought_process.append(f"Invoking 'rerank' on {len(candidates)} candidates -> selecting top {settings.rerank_top_k}")
            try:
                reranked_chunks = self.rerank_tool.execute(
                    query=search_query,
                    candidates=candidates,
                    top_k=settings.rerank_top_k,
                )
                tool_calls.append(
                    ToolCall(
                        tool_name="rerank",
                        arguments={"query": search_query, "candidates_count": len(candidates), "top_k": settings.rerank_top_k},
                        result=reranked_chunks,
                    )
                )
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

            # E. Merge and deduplicate by chunk_id
            added_count = 0
            for chunk in reranked_chunks:
                if chunk.chunk_id not in all_chunks_dict or chunk.rerank_score > all_chunks_dict[chunk.chunk_id].rerank_score:
                    all_chunks_dict[chunk.chunk_id] = chunk
                    added_count += 1

            thought_process.append(f"Hop {hop} added {added_count} new/updated chunks. Total unique context chunks: {len(all_chunks_dict)}.")

            # Form current context pool
            current_context = sorted(all_chunks_dict.values(), key=lambda x: -x.rerank_score)[:settings.rerank_top_k]

            # F. Sufficiency Check
            is_sufficient, missing_aspect = self.evaluate_sufficiency(query, current_context)
            hop_traces.append(
                HopTrace(
                    hop_index=hop,
                    query_used=search_query,
                    candidates_retrieved=len(candidates),
                    reranked_chunks_added=added_count,
                    is_sufficient=is_sufficient,
                    missing_aspect=missing_aspect,
                )
            )

            if is_sufficient:
                thought_process.append(f"Sufficiency check PASSED at Hop {hop}.")
                break
            else:
                thought_process.append(f"Sufficiency check: INSUFFICIENT. Missing aspect: '{missing_aspect}'.")

        # Step 3: Check context availability
        final_context = sorted(all_chunks_dict.values(), key=lambda x: -x.rerank_score)[:settings.rerank_top_k]

        if not final_context:
            thought_process.append("No matching context chunks found across all retrieval hops.")
            return AgentResponse(
                query=query,
                answer="Based on the available local documentation, no relevant information was found to answer this question.",
                retrieval_needed=True,
                tool_calls=tool_calls,
                thought_process=thought_process,
                hop_traces=hop_traces,
                hops_executed=executed_hops,
                rewritten_queries=rewritten_queries,
                metadata={"latency_sec": round(time.time() - start_time, 3)},
            )

        # Step 4: Synthesize Grounded Answer with LLM
        thought_process.append(f"Synthesizing final answer grounded on {len(final_context)} unique context chunks.")
        synthesis_prompt = build_synthesis_prompt(query=query, context_chunks=final_context)

        try:
            answer = self.llm.generate(
                prompt=synthesis_prompt,
                system=SYNTHESIS_SYSTEM_PROMPT,
                temperature=settings.agent_temperature,
            )
        except OllamaConnectionError as e:
            return self._handle_ollama_connection_error(query, e, thought_process, tool_calls, final_context)
        except Exception as e:
            logger.error(f"Error during LLM answer synthesis: {e}", exc_info=True)
            return AgentResponse(
                query=query,
                answer=f"Error generating answer: {e}",
                retrieval_needed=True,
                sources=[chunk.to_dict() for chunk in final_context],
                tool_calls=tool_calls,
                thought_process=thought_process,
                hop_traces=hop_traces,
                hops_executed=executed_hops,
                rewritten_queries=rewritten_queries,
                metadata={"error": str(e)},
            )

        elapsed_time = round(time.time() - start_time, 3)
        thought_process.append(f"Answer synthesized successfully in {elapsed_time}s across {executed_hops} hop(s).")

        sources = [chunk.to_dict() for chunk in final_context]

        return AgentResponse(
            query=query,
            answer=answer,
            retrieval_needed=True,
            sources=sources,
            tool_calls=tool_calls,
            thought_process=thought_process,
            hop_traces=hop_traces,
            hops_executed=executed_hops,
            rewritten_queries=rewritten_queries,
            metadata={
                "latency_sec": elapsed_time,
                "model": getattr(self.llm, "model", "mock"),
                "unique_chunks_used": len(final_context),
                "hops_executed": executed_hops,
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
