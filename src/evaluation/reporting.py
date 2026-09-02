import csv
import json
import logging
from pathlib import Path
from typing import Optional

from src.evaluation.models import EvaluationReport

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Handles formatted console printing and structured JSON/CSV export of evaluation runs."""

    @staticmethod
    def print_summary(report: EvaluationReport) -> None:
        """Print clean, formatted terminal summary tables of the evaluation run."""
        print("\n" + "=" * 80)
        print(f"HYBRID-AGENTIC RAG EVALUATION REPORT [{report.mode.upper()} MODE]")
        print("=" * 80)
        print(f"Run ID        : {report.run_id}")
        print(f"Timestamp     : {report.timestamp}")
        print(f"Git Commit    : {report.git_commit}")
        print(f"Dataset Hash  : {report.dataset_hash} (v{report.dataset_version})")
        print(f"Duration      : {report.execution_duration_sec:.3f} s")
        print(f"Total Queries : {len(report.per_query_results)}")
        print("-" * 80)

        # 1. Retrieval Comparison Table
        if report.retrieval_comparison:
            print("\n1. RETRIEVAL PIPELINE COMPARISON (K = 1, 3, 5, 10)")
            print(f"{'Pipeline':<20} | {'Hit@1':<7} | {'Hit@5':<7} | {'Rec@5':<7} | {'Prec@5':<7} | {'MRR':<7} | {'nDCG@5':<7}")
            print("-" * 80)
            for name, m in report.retrieval_comparison.items():
                h1 = m.hit_rate.get(1, 0.0)
                h5 = m.hit_rate.get(5, 0.0)
                r5 = m.recall.get(5, 0.0)
                p5 = m.precision.get(5, 0.0)
                mrr = m.mrr
                nd5 = m.ndcg.get(5, 0.0)
                print(f"{name:<20} | {h1:<7.4f} | {h5:<7.4f} | {r5:<7.4f} | {p5:<7.4f} | {mrr:<7.4f} | {nd5:<7.4f}")

        # 2. Reranker Improvement Table
        if report.reranker_analysis:
            r = report.reranker_analysis
            print("\n2. CROSS-ENCODER RERANKER IMPROVEMENTS (OVER HYBRID RRF)")
            print(f" - Pre-Rerank MRR  : {r.pre_rerank_mrr:.4f}  -->  Post-Rerank MRR  : {r.post_rerank_mrr:.4f}  (Delta: {r.delta_mrr:+.4f})")
            print(f" - Pre-Rerank Hit@1: {r.pre_rerank_hit_1:.4f}  -->  Post-Rerank Hit@1: {r.post_rerank_hit_1:.4f}  (Delta: {r.delta_hit_1:+.4f})")
            print(f" - Pre-Rerank nDCG5: {r.pre_rerank_ndcg_5:.4f}  -->  Post-Rerank nDCG5: {r.post_rerank_ndcg_5:.4f}  (Delta: {r.delta_ndcg_5:+.4f})")
            print(f" - Promotion Rate  : {r.promotion_rate * 100:.1f}% of queries saw relevant chunks move upward")
            print(f" - Avg Rank Shift  : {r.average_rank_shift:+.2f} positions")

        # 3. CRAG Analysis
        if report.crag_analysis:
            c = report.crag_analysis
            print("\n3. CORRECTIVE RAG (CRAG) ORCHESTRATION")
            print(f" - Correction Trigger Rate : {c.correction_trigger_rate * 100:.1f}% ({c.correction_trigger_count}/{c.total_queries})")
            print(f" - Correction Success Rate : {c.successful_correction_rate * 100:.1f}% ({c.successful_correction_count}/{c.correction_trigger_count or 1})")
            print(f" - Refusal Accuracy        : {c.refusal_accuracy * 100:.1f}%")
            print(f" - Average Hops (Max 2)    : {c.average_hops:.2f}")
            print(f" - Grade Distribution      : {c.grade_distribution}")

        # 4. Answer Analysis
        if report.answer_analysis:
            a = report.answer_analysis
            print("\n4. ANSWER SYNTHESIS & GROUNDING")
            print(f" - Mean Aspect Coverage          : {a.mean_aspect_coverage * 100:.1f}%")
            print(f" - Source Attribution Coverage   : {a.mean_source_attribution_coverage * 100:.1f}%")
            print(f" - Refusal Accuracy              : {a.refusal_accuracy * 100:.1f}%")

        print("\n" + "=" * 80 + "\n")

    @staticmethod
    def save_json(report: EvaluationReport, output_dir: Path) -> Path:
        """Save report to JSON file and update latest.json pointer."""
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{report.run_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Also write latest.json
        latest_path = output_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return file_path

    @staticmethod
    def save_csv(report: EvaluationReport, output_dir: Path) -> Path:
        """Save per-query evaluation results to CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{report.run_id}.csv"

        fieldnames = [
            "query_id",
            "category",
            "query",
            "dense_mrr",
            "sparse_mrr",
            "hybrid_mrr",
            "reranked_mrr",
            "aspect_coverage",
            "refusal_correct",
            "crag_grade",
            "is_corrected",
            "hops_executed",
            "latency_ms",
        ]

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in report.per_query_results:
                row = {
                    "query_id": r.query_id,
                    "category": r.category,
                    "query": r.query,
                    "dense_mrr": round(r.dense_mrr, 4),
                    "sparse_mrr": round(r.sparse_mrr, 4),
                    "hybrid_mrr": round(r.hybrid_mrr, 4),
                    "reranked_mrr": round(r.reranked_mrr, 4),
                    "aspect_coverage": round(r.aspect_coverage, 4),
                    "refusal_correct": r.refusal_correct,
                    "crag_grade": r.crag_grade or "N/A",
                    "is_corrected": r.is_corrected,
                    "hops_executed": r.hops_executed,
                    "latency_ms": round(r.latency_ms, 2),
                }
                writer.writerow(row)

        return file_path
