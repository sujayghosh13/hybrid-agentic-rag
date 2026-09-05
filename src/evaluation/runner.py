import datetime
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.agent import LocalQwenAgent
from src.config import settings
from src.evaluation.answer_eval import AnswerEvaluator
from src.evaluation.crag_eval import CRAGEvaluator
from src.evaluation.dataset import load_benchmark_dataset, validate_benchmark_dataset
from src.evaluation.metrics import reciprocal_rank
from src.evaluation.models import EvaluationReport, PerQueryEvalResult
from src.evaluation.reranker_eval import RerankerEvaluator
from src.evaluation.retrieval_eval import RetrievalEvaluator

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Retrieve current short git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class EvaluationRunner:
    """Master orchestrator for running FAST or FULL evaluation pipelines."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        corpus_chunks_path: Optional[Path] = None,
    ):
        self.dataset_path = dataset_path or Path("data/evaluation/benchmark_dataset.json")
        self.corpus_chunks_path = corpus_chunks_path or Path("data/processed/chunks.jsonl")

    def run(self, mode: str = "fast") -> EvaluationReport:
        """Execute evaluation in either 'fast' or 'full' mode."""
        start_time = time.perf_counter()
        run_id = f"eval_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        git_commit = get_git_commit()

        logger.info(f"Starting Evaluation Run: {run_id} [Mode: {mode}]")

        # 1. Load and validate benchmark dataset
        queries, metadata = load_benchmark_dataset(self.dataset_path)
        is_valid, validation_errors = validate_benchmark_dataset(queries, self.corpus_chunks_path)
        if not is_valid:
            logger.warning(f"Benchmark dataset validation warnings: {validation_errors}")

        # Extract corpus chunk IDs for attribution checks
        corpus_ids = set()
        if self.corpus_chunks_path.exists():
            import json
            with open(self.corpus_chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            corpus_ids.add(json.loads(line)["id"])
                        except Exception:
                            pass

        environment = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "embedding_model": settings.embedding_model_name,
            "reranker_model": settings.reranker_model_name,
            "ollama_model": settings.ollama_model,
            "crag_enabled": settings.crag_enabled,
            "total_corpus_chunks": len(corpus_ids),
        }

        # 2. Execute Retrieval Evaluation (always executed across both modes)
        retrieval_evaluator = RetrievalEvaluator()
        retrieval_maps = retrieval_evaluator.run_all_retrievers(queries, top_k=10)

        retrieval_comparison = {}
        for name, r_map in retrieval_maps.items():
            retrieval_comparison[name] = retrieval_evaluator.evaluate_retriever(
                queries=queries,
                pipeline_name=name,
                retrieved_ids_map=r_map,
            )

        # 3. Execute Reranker Evaluation
        reranker_evaluator = RerankerEvaluator()
        reranker_metrics = reranker_evaluator.evaluate_reranker(
            queries=queries,
            pre_rerank_map=retrieval_maps["hybrid_rrf"],
            post_rerank_map=retrieval_maps["hybrid_reranked"],
        )

        crag_metrics = None
        answer_metrics = None
        per_query_results: List[PerQueryEvalResult] = []
        failures: List[Dict[str, Any]] = []

        if mode == "full":
            logger.info("Executing FULL mode with LocalQwenAgent orchestration...")
            from src.agent.tools import HybridSearchTool, RerankTool
            search_tool = HybridSearchTool(retriever=retrieval_evaluator.hybrid_retriever)
            rerank_tool = RerankTool(reranker=retrieval_evaluator.reranker)
            agent = LocalQwenAgent(
                hybrid_search_tool=search_tool,
                rerank_tool=rerank_tool,
            )
            agent_responses = []

            for q in queries:
                q_t0 = time.perf_counter()
                error_msg = None
                try:
                    resp = agent.run(q.query)
                except Exception as e:
                    logger.error(f"Error evaluating query '{q.id}': {e}", exc_info=True)
                    error_msg = str(e)
                    resp = None
                    failures.append({"query_id": q.id, "error": error_msg})
                q_t1 = time.perf_counter()

                if resp is not None:
                    agent_responses.append(resp)
                    d_mrr = reciprocal_rank(retrieval_maps["dense"].get(q.id, []), q.relevant_chunk_ids)
                    s_mrr = reciprocal_rank(retrieval_maps["sparse"].get(q.id, []), q.relevant_chunk_ids)
                    h_mrr = reciprocal_rank(retrieval_maps["hybrid_rrf"].get(q.id, []), q.relevant_chunk_ids)
                    r_mrr = reciprocal_rank(retrieval_maps["hybrid_reranked"].get(q.id, []), q.relevant_chunk_ids)

                    is_refusal = "insufficient evidence" in resp.answer.lower()
                    ref_correct = (is_refusal and q.expected_refusal) or (not is_refusal and not q.expected_refusal)

                    from src.evaluation.metrics import aspect_coverage_score
                    cov = aspect_coverage_score(resp.answer, q.expected_aspects) if not q.expected_refusal else 1.0

                    per_query_results.append(
                        PerQueryEvalResult(
                            query_id=q.id,
                            query=q.query,
                            category=q.category,
                            dense_hits=retrieval_maps["dense"].get(q.id, [])[:5],
                            sparse_hits=retrieval_maps["sparse"].get(q.id, [])[:5],
                            hybrid_hits=retrieval_maps["hybrid_rrf"].get(q.id, [])[:5],
                            reranked_hits=retrieval_maps["hybrid_reranked"].get(q.id, [])[:5],
                            relevant_chunks=q.relevant_chunk_ids,
                            dense_mrr=d_mrr,
                            sparse_mrr=s_mrr,
                            hybrid_mrr=h_mrr,
                            reranked_mrr=r_mrr,
                            aspect_coverage=cov,
                            refusal_correct=ref_correct,
                            crag_grade=(
                                resp.final_evidence_grade.value
                                if hasattr(resp.final_evidence_grade, "value")
                                else (str(resp.final_evidence_grade) if resp.final_evidence_grade else None)
                            ),
                            is_corrected=resp.is_corrected,
                            hops_executed=resp.hops_executed,
                            latency_ms=(q_t1 - q_t0) * 1000.0,
                            error=error_msg,
                        )
                    )

            # CRAG & Answer Evaluators
            crag_evaluator = CRAGEvaluator()
            crag_metrics = crag_evaluator.evaluate_crag(queries, agent_responses)

            answer_evaluator = AnswerEvaluator(corpus_chunk_ids=corpus_ids)
            answer_metrics = answer_evaluator.evaluate_answers(queries, agent_responses)

        else:
            # FAST mode: Populate per-query retrieval metrics without LLM generation
            for q in queries:
                d_mrr = reciprocal_rank(retrieval_maps["dense"].get(q.id, []), q.relevant_chunk_ids)
                s_mrr = reciprocal_rank(retrieval_maps["sparse"].get(q.id, []), q.relevant_chunk_ids)
                h_mrr = reciprocal_rank(retrieval_maps["hybrid_rrf"].get(q.id, []), q.relevant_chunk_ids)
                r_mrr = reciprocal_rank(retrieval_maps["hybrid_reranked"].get(q.id, []), q.relevant_chunk_ids)

                per_query_results.append(
                    PerQueryEvalResult(
                        query_id=q.id,
                        query=q.query,
                        category=q.category,
                        dense_hits=retrieval_maps["dense"].get(q.id, [])[:5],
                        sparse_hits=retrieval_maps["sparse"].get(q.id, [])[:5],
                        hybrid_hits=retrieval_maps["hybrid_rrf"].get(q.id, [])[:5],
                        reranked_hits=retrieval_maps["hybrid_reranked"].get(q.id, [])[:5],
                        relevant_chunks=q.relevant_chunk_ids,
                        dense_mrr=d_mrr,
                        sparse_mrr=s_mrr,
                        hybrid_mrr=h_mrr,
                        reranked_mrr=r_mrr,
                        aspect_coverage=1.0 if q.expected_refusal else 0.0,
                        refusal_correct=True,
                        crag_grade=None,
                        is_corrected=False,
                        hops_executed=1,
                        latency_ms=0.0,
                        error=None,
                    )
                )

        end_time = time.perf_counter()
        duration_sec = end_time - start_time

        return EvaluationReport(
            run_id=run_id,
            timestamp=timestamp,
            mode=mode,
            git_commit=git_commit,
            dataset_version=metadata.get("version", "1.0.0"),
            dataset_hash=metadata.get("dataset_hash", "unknown"),
            execution_duration_sec=duration_sec,
            environment=environment,
            retrieval_comparison=retrieval_comparison,
            reranker_analysis=reranker_metrics,
            crag_analysis=crag_metrics,
            answer_analysis=answer_metrics,
            per_query_results=per_query_results,
            failures=failures,
        )
