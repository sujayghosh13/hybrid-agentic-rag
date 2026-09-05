import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.agent.llm import BaseLLMClient, OllamaClient, OllamaConnectionError, OllamaError, call_llm_generate
from src.agent.models import AgentResponse, HopTrace, ToolCall
from src.agent.prompts import (
    REWRITE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SUFFICIENCY_SYSTEM_PROMPT,
    build_rewrite_prompt,
    build_sufficiency_prompt,
    build_synthesis_prompt,
)
from src.agent.tools import BaseTool, HybridSearchTool, RerankTool, ToolRegistry
from src.config import settings
from src.correction.corrective_action import CorrectiveActionEngine
from src.correction.evaluator import EvidenceEvaluator
from src.correction.models import CorrectionTrace, EvidenceEvaluation, EvidenceGrade
from src.reranking.models import RerankedResult

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are an offline-first technical AI assistant.
Your job is to provide accurate, factual, and strictly grounded answers based ONLY on the provided documentation context.

STRICT GROUNDING & FACTUALITY RULES:
1. Strict Evidence Grounding: Base your answer strictly and exclusively on the retrieved context chunks. Never infer, extrapolate, or invent networking behavior, routing mechanisms, or gateway capabilities that are not explicitly documented in the retrieved text.
2. Distinguish Network Scopes: Clearly distinguish between:
   - default Docker behavior (e.g., default bridge network isolation; containers on the default bridge can communicate via IP address only, not by container name)
   - user-defined network behavior (e.g., scoped isolation; containers on the same user-defined network can communicate via container names or IP addresses)
   - behavior requiring explicit configuration (e.g., connecting a container to multiple networks using `docker network connect`, publishing ports)
3. Cross-Network Isolation: Containers connected to different/separate networks CANNOT communicate directly. Docker enforces network isolation between distinct networks. Do NOT state or imply that containers on separate networks can communicate directly, nor that Docker routes traffic between separate networks via default gateways or `--gw-priority` (default gateways are only used for destinations outside a container's directly connected networks).
4. Legacy Flags: Do not use or suggest legacy '--link' as a general solution unless the retrieved evidence specifically requires it.
5. Insufficient Evidence: If the retrieved evidence does not explain a mechanism for containers on different networks to communicate other than attaching to the same network or publishing ports, explicitly state that the retrieved evidence does not provide for direct cross-network routing.
6. Technical Precision: Be clear, concise, direct, and technically precise without guessing or speculating.
7. Output Format: Answer directly, clearly, and concisely. Do not output chain-of-thought scratchpad text or repeat introductory statements.
"""


class LocalQwenAgent:
    """Local offline-first Agent powered by Qwen3 via Ollama with Corrective RAG (CRAG)."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        hybrid_search_tool: Optional[HybridSearchTool] = None,
        rerank_tool: Optional[RerankTool] = None,
        registry: Optional[ToolRegistry] = None,
        evidence_evaluator: Optional[EvidenceEvaluator] = None,
        corrective_action_engine: Optional[CorrectiveActionEngine] = None,
    ):
        self.llm = llm_client or OllamaClient()
        self.search_tool = hybrid_search_tool or HybridSearchTool()
        self.rerank_tool = rerank_tool or RerankTool()

        self.registry = registry or ToolRegistry()
        self.registry.register(self.search_tool)
        self.registry.register(self.rerank_tool)

        self.evaluator = evidence_evaluator or EvidenceEvaluator(llm_client=self.llm)
        self.corrective_engine = corrective_action_engine or CorrectiveActionEngine(llm_client=self.llm)

        # In-memory LRU / lookup caches for fast sub-millisecond repeated query processing
        self._routing_cache: Dict[str, bool] = {}
        self._rewrite_cache: Dict[Tuple[str, Optional[str]], str] = {}

    def should_retrieve(self, query: str) -> bool:
        """Decide if local technical documentation retrieval is required for the query."""
        clean_q = query.strip().lower()

        # Cache check
        if clean_q in self._routing_cache:
            return self._routing_cache[clean_q]

        # Conversational / greeting heuristic filter
        if clean_q in ("hello", "hi", "hey", "who are you?", "what can you do?", "thanks", "thank you"):
            self._routing_cache[clean_q] = False
            return False

        # Technical domain terms fast-path
        tech_keywords = (
            "docker", "kubernetes", "k8s", "bridge", "network", "networking",
            "driver", "container", "pod", "pods", "service", "deployment",
            "daemon", "volume", "port", "ingress", "cluster", "workload",
            "image", "config", "ip", "subnet", "gateway", "how", "what", "why"
        )
        if any(kw in clean_q for kw in tech_keywords):
            self._routing_cache[clean_q] = True
            return True

        try:
            routing_decision = call_llm_generate(
                self.llm,
                prompt=f"User Query: {query}\nDecision:",
                system=ROUTER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=10,
            )
            decision = routing_decision.strip().upper()
            logger.info(f"[Agent Routing] Query: '{query}' -> Decision: '{decision}'")
            res = "RETRIEVE" in decision
            self._routing_cache[clean_q] = res
            return res
        except Exception as e:
            logger.warning(f"Routing classifier encountered error: {e}. Defaulting to RETRIEVE.")
            self._routing_cache[clean_q] = True
            return True

    def rewrite_query(self, query: str, missing_aspect: Optional[str] = None) -> str:
        """Rewrite a user query into concise, high-signal retrieval keywords."""
        if not settings.query_rewriter_enabled:
            return query

        cache_key = (query.strip().lower(), missing_aspect.strip().lower() if missing_aspect else None)
        if cache_key in self._rewrite_cache:
            return self._rewrite_cache[cache_key]

        prompt = build_rewrite_prompt(query, missing_aspect=missing_aspect)
        try:
            rewritten = call_llm_generate(
                self.llm,
                prompt=prompt,
                system=REWRITE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=40,
                stop=["\n"],
            )
            cleaned = rewritten.strip().strip('"').strip("'").split("\n")[0].strip()
            if cleaned and len(cleaned) > 2:
                logger.info(f"[Query Rewriter] '{query}' -> '{cleaned}' (missing: '{missing_aspect}')")
                self._rewrite_cache[cache_key] = cleaned
                return cleaned
        except Exception as e:
            logger.warning(f"Query rewriter encountered error: {e}. Falling back to original query.")

        self._rewrite_cache[cache_key] = query
        return query

    def route_and_rewrite(self, query: str) -> Tuple[bool, str]:
        """Unified router and rewriter optimization pass.
        
        Determines retrieval requirement and optimal retrieval query in a single unified step,
        leveraging both routing and rewrite fast-paths and caches.
        """
        clean_q = query.strip()
        needs_retrieval = self.should_retrieve(clean_q)
        if not needs_retrieval:
            return False, clean_q
        rewritten = self.rewrite_query(clean_q)
        return True, rewritten

    def _calculate_adaptive_max_tokens(self, query: str, context_chunks: List[RerankedResult]) -> int:
        """Dynamically compute max_tokens based on query complexity and retrieved context size."""
        # Base budget for concise answers
        budget = 400
        # Multi-part or complex questions get additional generation headroom
        lower_q = query.lower()
        if any(term in lower_q for term in ("difference", "compare", "steps", "explain how", "how to", "why", "and", "both")):
            budget += 150
        # More context blocks warrant slightly higher token limit for comprehensive coverage
        if len(context_chunks) >= 3:
            budget += 100
        return min(budget, 650)

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
            eval_response = call_llm_generate(
                self.llm,
                prompt=prompt,
                system=SUFFICIENCY_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=60,
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
        """Run the full agentic CRAG workflow with unified global retrieval budget (max 2 hops)."""
        start_time = time.time()
        thought_process: List[str] = []
        tool_calls: List[ToolCall] = []
        hop_traces: List[HopTrace] = []
        correction_traces: List[CorrectionTrace] = []
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

        # Step 1: Query Routing & Rewriting (Unified pass with cache)
        retrieval_needed, search_query = self.route_and_rewrite(query)

        if not retrieval_needed:
            thought_process.append("Query classified as conversational/direct. Generating direct answer.")
            try:
                direct_answer = call_llm_generate(
                    self.llm,
                    prompt=query,
                    system="You are a helpful and polite technical AI assistant.",
                    max_tokens=800,
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

        # Step 2: Unified Global Retrieval Budget Lifecycle (max_hops <= 2)
        all_chunks_dict: Dict[str, RerankedResult] = {}
        executed_queries: List[str] = []
        max_retrieval_hops = min(settings.agent_max_hops, 2)
        total_retrieval_hops = 0

        # --- GLOBAL RETRIEVAL HOP 1: Initial Retrieval & CRAG Evaluation ---
        total_retrieval_hops += 1
        thought_process.append(f"--- Global Retrieval Hop 1/{max_retrieval_hops}: Initial Retrieval ---")

        rewritten_queries.append(search_query)
        executed_queries.append(search_query)

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

        reranked_chunks: List[RerankedResult] = []
        if candidates:
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

        added_count_1 = 0
        for chunk in reranked_chunks:
            if chunk.chunk_id not in all_chunks_dict or chunk.rerank_score > all_chunks_dict[chunk.chunk_id].rerank_score:
                all_chunks_dict[chunk.chunk_id] = chunk
                added_count_1 += 1

        thought_process.append(f"Hop 1 added {added_count_1} new/updated chunks. Total unique context chunks: {len(all_chunks_dict)}.")

        current_context = sorted(all_chunks_dict.values(), key=lambda x: -x.rerank_score)[:settings.rerank_top_k]

        # Evaluate Evidence Quality (Deterministic fast-path + LLM)
        eval_result = self.evaluator.evaluate(query, current_context)
        thought_process.append(
            f"Evidence Quality Grade: {eval_result.grade.value} | Reason: {eval_result.reason or 'N/A'} | Missing: {eval_result.missing_aspect or 'None'}"
        )

        hop_traces.append(
            HopTrace(
                hop_index=1,
                query_used=search_query,
                candidates_retrieved=len(candidates),
                reranked_chunks_added=added_count_1,
                is_sufficient=(eval_result.grade == EvidenceGrade.GOOD),
                missing_aspect=eval_result.missing_aspect,
            )
        )
        correction_traces.append(
            CorrectionTrace(
                hop_index=1,
                query_used=search_query,
                evidence_grade=eval_result.grade,
                missing_aspect=eval_result.missing_aspect,
                reason=eval_result.reason,
                action_taken="PROCEED_TO_SYNTHESIS" if eval_result.grade == EvidenceGrade.GOOD else f"TRIGGER_CORRECTION_{eval_result.grade.value}",
                candidates_retrieved=len(candidates),
                reranked_chunks_added=added_count_1,
            )
        )

        # --- GLOBAL RETRIEVAL HOP 2: Corrective Retrieval (If PARTIAL or BAD & Budget Permits) ---
        if eval_result.grade in (EvidenceGrade.PARTIAL, EvidenceGrade.BAD) and total_retrieval_hops < max_retrieval_hops and settings.crag_enabled:
            corrective_query = self.corrective_engine.generate_corrective_query(query, eval_result)

            if corrective_query in executed_queries:
                thought_process.append(f"Corrective query '{corrective_query}' already searched. Breaking loop to prevent duplicates.")
            else:
                total_retrieval_hops += 1
                thought_process.append(f"--- Global Retrieval Hop 2/{max_retrieval_hops}: Corrective Retrieval ({eval_result.grade.value}) ---")
                thought_process.append(f"Executing corrective query: '{corrective_query}'")

                rewritten_queries.append(corrective_query)
                executed_queries.append(corrective_query)

                try:
                    corr_candidates = self.search_tool.execute(query=corrective_query, top_k=settings.rerank_candidates_count)
                    tool_calls.append(
                        ToolCall(
                            tool_name="hybrid_search",
                            arguments={"query": corrective_query, "top_k": settings.rerank_candidates_count},
                            result=corr_candidates,
                        )
                    )
                    thought_process.append(f"Corrective hybrid search returned {len(corr_candidates)} candidate chunks.")
                except Exception as e:
                    logger.error(f"Error in corrective hybrid search: {e}", exc_info=True)
                    corr_candidates = []

                corr_reranked: List[RerankedResult] = []
                if corr_candidates:
                    try:
                        corr_reranked = self.rerank_tool.execute(
                            query=corrective_query,
                            candidates=corr_candidates,
                            top_k=settings.rerank_top_k,
                        )
                        tool_calls.append(
                            ToolCall(
                                tool_name="rerank",
                                arguments={"query": corrective_query, "candidates_count": len(corr_candidates), "top_k": settings.rerank_top_k},
                                result=corr_reranked,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error in corrective rerank: {e}", exc_info=True)

                added_count_2 = 0
                for chunk in corr_reranked:
                    if chunk.chunk_id not in all_chunks_dict or chunk.rerank_score > all_chunks_dict[chunk.chunk_id].rerank_score:
                        all_chunks_dict[chunk.chunk_id] = chunk
                        added_count_2 += 1

                thought_process.append(f"Hop 2 added {added_count_2} new/updated chunks. Total unique context chunks: {len(all_chunks_dict)}.")

                current_context = sorted(all_chunks_dict.values(), key=lambda x: -x.rerank_score)[:settings.rerank_top_k]

                # Re-evaluate evidence quality after corrective retrieval
                final_eval = self.evaluator.evaluate(query, current_context)
                thought_process.append(
                    f"Post-Correction Evidence Grade: {final_eval.grade.value} | Reason: {final_eval.reason or 'N/A'} | Missing: {final_eval.missing_aspect or 'None'}"
                )

                hop_traces.append(
                    HopTrace(
                        hop_index=2,
                        query_used=corrective_query,
                        candidates_retrieved=len(corr_candidates),
                        reranked_chunks_added=added_count_2,
                        is_sufficient=(final_eval.grade == EvidenceGrade.GOOD),
                        missing_aspect=final_eval.missing_aspect,
                    )
                )
                correction_traces.append(
                    CorrectionTrace(
                        hop_index=2,
                        query_used=corrective_query,
                        evidence_grade=final_eval.grade,
                        missing_aspect=final_eval.missing_aspect,
                        reason=final_eval.reason,
                        action_taken="PROCEED_TO_SYNTHESIS" if final_eval.grade == EvidenceGrade.GOOD else "SYNTHESIS_WITH_INCOMPLETE_EVIDENCE",
                        candidates_retrieved=len(corr_candidates),
                        reranked_chunks_added=added_count_2,
                    )
                )
                eval_result = final_eval

        # Step 3: Final Context Pool & Anti-Hallucination Refusal Check
        final_context = sorted(all_chunks_dict.values(), key=lambda x: -x.rerank_score)[:settings.rerank_top_k]

        if not final_context or eval_result.grade == EvidenceGrade.BAD:
            thought_process.append("Evidence quality is BAD / No usable documentation found. Refusing to hallucinate.")
            return AgentResponse(
                query=query,
                answer="Based on the available local technical documentation, there is insufficient evidence to answer this question.",
                retrieval_needed=True,
                sources=[chunk.to_dict() for chunk in final_context],
                tool_calls=tool_calls,
                thought_process=thought_process,
                hop_traces=hop_traces,
                hops_executed=total_retrieval_hops,
                rewritten_queries=rewritten_queries,
                correction_traces=correction_traces,
                final_evidence_grade=eval_result.grade.value,
                is_corrected=(len(correction_traces) > 1),
                metadata={"latency_sec": round(time.time() - start_time, 3), "refusal": True},
            )

        # Step 4: Synthesize Final Grounded Answer with LLM
        thought_process.append(f"Synthesizing final answer grounded strictly on {len(final_context)} unique context chunks.")
        synthesis_prompt = build_synthesis_prompt(query=query, context_chunks=final_context)

        adaptive_tokens = self._calculate_adaptive_max_tokens(query, final_context)
        try:
            answer = call_llm_generate(
                self.llm,
                prompt=synthesis_prompt,
                system=SYNTHESIS_SYSTEM_PROMPT,
                temperature=settings.agent_temperature,
                max_tokens=adaptive_tokens,
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
                hops_executed=total_retrieval_hops,
                rewritten_queries=rewritten_queries,
                correction_traces=correction_traces,
                final_evidence_grade=eval_result.grade.value,
                is_corrected=(len(correction_traces) > 1),
                metadata={"error": str(e)},
            )

        elapsed_time = round(time.time() - start_time, 3)
        thought_process.append(f"Answer synthesized successfully in {elapsed_time}s across {total_retrieval_hops} global hop(s).")

        sources = [chunk.to_dict() for chunk in final_context]

        return AgentResponse(
            query=query,
            answer=answer,
            retrieval_needed=True,
            sources=sources,
            tool_calls=tool_calls,
            thought_process=thought_process,
            hop_traces=hop_traces,
            hops_executed=total_retrieval_hops,
            rewritten_queries=rewritten_queries,
            correction_traces=correction_traces,
            final_evidence_grade=eval_result.grade.value,
            is_corrected=(len(correction_traces) > 1),
            metadata={
                "latency_sec": elapsed_time,
                "model": getattr(self.llm, "model", "mock"),
                "unique_chunks_used": len(final_context),
                "hops_executed": total_retrieval_hops,
                "final_evidence_grade": eval_result.grade.value,
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
