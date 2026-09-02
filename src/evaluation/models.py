import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkQuery:
    """Represents a single verified benchmark query."""

    id: str
    query: str
    category: str = "general"
    relevant_doc_files: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    expected_aspects: List[str] = field(default_factory=list)
    ground_truth_answer: str = ""
    expected_refusal: bool = False
    min_relevant_chunks: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQuery":
        return cls(
            id=data["id"],
            query=data["query"],
            category=data.get("category", "general"),
            relevant_doc_files=data.get("relevant_doc_files", []),
            relevant_chunk_ids=data.get("relevant_chunk_ids", []),
            expected_aspects=data.get("expected_aspects", []),
            ground_truth_answer=data.get("ground_truth_answer", ""),
            expected_refusal=data.get("expected_refusal", False),
            min_relevant_chunks=data.get("min_relevant_chunks", 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category,
            "relevant_doc_files": self.relevant_doc_files,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "expected_aspects": self.expected_aspects,
            "ground_truth_answer": self.ground_truth_answer,
            "expected_refusal": self.expected_refusal,
            "min_relevant_chunks": self.min_relevant_chunks,
        }


@dataclass
class RetrievalMetrics:
    """Calculated metrics for a specific retriever at standardized K values (1, 3, 5, 10)."""

    hit_rate: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    precision: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit_rate": {f"@{k}": round(v, 4) for k, v in self.hit_rate.items()},
            "recall": {f"@{k}": round(v, 4) for k, v in self.recall.items()},
            "precision": {f"@{k}": round(v, 4) for k, v in self.precision.items()},
            "mrr": round(self.mrr, 4),
            "ndcg": {f"@{k}": round(v, 4) for k, v in self.ndcg.items()},
        }


@dataclass
class RerankerMetrics:
    """Metrics measuring the impact of Cross-Encoder reranking over Hybrid RRF."""

    pre_rerank_mrr: float = 0.0
    post_rerank_mrr: float = 0.0
    delta_mrr: float = 0.0
    pre_rerank_hit_1: float = 0.0
    post_rerank_hit_1: float = 0.0
    delta_hit_1: float = 0.0
    pre_rerank_ndcg_5: float = 0.0
    post_rerank_ndcg_5: float = 0.0
    delta_ndcg_5: float = 0.0
    promotion_rate: float = 0.0
    average_rank_shift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pre_rerank_mrr": round(self.pre_rerank_mrr, 4),
            "post_rerank_mrr": round(self.post_rerank_mrr, 4),
            "delta_mrr": round(self.delta_mrr, 4),
            "pre_rerank_hit_1": round(self.pre_rerank_hit_1, 4),
            "post_rerank_hit_1": round(self.post_rerank_hit_1, 4),
            "delta_hit_1": round(self.delta_hit_1, 4),
            "pre_rerank_ndcg_5": round(self.pre_rerank_ndcg_5, 4),
            "post_rerank_ndcg_5": round(self.post_rerank_ndcg_5, 4),
            "delta_ndcg_5": round(self.delta_ndcg_5, 4),
            "promotion_rate": round(self.promotion_rate, 4),
            "average_rank_shift": round(self.average_rank_shift, 4),
        }


@dataclass
class CRAGMetrics:
    """Metrics evaluating Corrective RAG behaviors and transitions."""

    total_queries: int = 0
    correction_trigger_count: int = 0
    correction_trigger_rate: float = 0.0
    successful_correction_count: int = 0
    successful_correction_rate: float = 0.0
    grade_distribution: Dict[str, int] = field(default_factory=dict)
    mean_evidence_gain: float = 0.0
    refusal_accuracy: float = 0.0
    average_hops: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "correction_trigger_count": self.correction_trigger_count,
            "correction_trigger_rate": round(self.correction_trigger_rate, 4),
            "successful_correction_count": self.successful_correction_count,
            "successful_correction_rate": round(self.successful_correction_rate, 4),
            "grade_distribution": self.grade_distribution,
            "mean_evidence_gain": round(self.mean_evidence_gain, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "average_hops": round(self.average_hops, 4),
        }


@dataclass
class AnswerMetrics:
    """Metrics evaluating the synthesized response and grounding."""

    mean_aspect_coverage: float = 0.0
    refusal_accuracy: float = 0.0
    mean_source_attribution_coverage: float = 0.0
    mean_faithfulness: Optional[float] = None
    mean_relevance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "mean_aspect_coverage": round(self.mean_aspect_coverage, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "mean_source_attribution_coverage": round(self.mean_source_attribution_coverage, 4),
        }
        if self.mean_faithfulness is not None:
            res["mean_faithfulness"] = round(self.mean_faithfulness, 4)
        if self.mean_relevance is not None:
            res["mean_relevance"] = round(self.mean_relevance, 4)
        return res


@dataclass
class PerQueryEvalResult:
    """Detailed evaluation result for an individual benchmark query."""

    query_id: str
    query: str
    category: str
    dense_hits: List[str] = field(default_factory=list)
    sparse_hits: List[str] = field(default_factory=list)
    hybrid_hits: List[str] = field(default_factory=list)
    reranked_hits: List[str] = field(default_factory=list)
    relevant_chunks: List[str] = field(default_factory=list)
    dense_mrr: float = 0.0
    sparse_mrr: float = 0.0
    hybrid_mrr: float = 0.0
    reranked_mrr: float = 0.0
    aspect_coverage: float = 0.0
    refusal_correct: bool = True
    crag_grade: Optional[str] = None
    is_corrected: bool = False
    hops_executed: int = 1
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "dense_hits": self.dense_hits,
            "sparse_hits": self.sparse_hits,
            "hybrid_hits": self.hybrid_hits,
            "reranked_hits": self.reranked_hits,
            "relevant_chunks": self.relevant_chunks,
            "dense_mrr": round(self.dense_mrr, 4),
            "sparse_mrr": round(self.sparse_mrr, 4),
            "hybrid_mrr": round(self.hybrid_mrr, 4),
            "reranked_mrr": round(self.reranked_mrr, 4),
            "aspect_coverage": round(self.aspect_coverage, 4),
            "refusal_correct": self.refusal_correct,
            "crag_grade": self.crag_grade,
            "is_corrected": self.is_corrected,
            "hops_executed": self.hops_executed,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """Master evaluation report containing all pipeline comparisons, metrics, and metadata."""

    run_id: str
    timestamp: str
    mode: str
    git_commit: str
    dataset_version: str
    dataset_hash: str
    execution_duration_sec: float
    environment: Dict[str, Any]
    retrieval_comparison: Dict[str, RetrievalMetrics] = field(default_factory=dict)
    reranker_analysis: Optional[RerankerMetrics] = None
    crag_analysis: Optional[CRAGMetrics] = None
    answer_analysis: Optional[AnswerMetrics] = None
    per_query_results: List[PerQueryEvalResult] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "git_commit": self.git_commit,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "execution_duration_sec": round(self.execution_duration_sec, 3),
            "environment": self.environment,
            "retrieval_comparison": {
                k: v.to_dict() for k, v in self.retrieval_comparison.items()
            },
            "reranker_analysis": self.reranker_analysis.to_dict() if self.reranker_analysis else None,
            "crag_analysis": self.crag_analysis.to_dict() if self.crag_analysis else None,
            "answer_analysis": self.answer_analysis.to_dict() if self.answer_analysis else None,
            "per_query_results": [r.to_dict() for r in self.per_query_results],
            "failures": self.failures,
        }
