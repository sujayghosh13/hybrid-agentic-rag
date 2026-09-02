import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.models import AgentResponse, HopTrace
from src.correction.models import CorrectionTrace, EvidenceGrade
from src.evaluation.answer_eval import AnswerEvaluator
from src.evaluation.crag_eval import CRAGEvaluator
from src.evaluation.dataset import load_benchmark_dataset, validate_benchmark_dataset
from src.evaluation.metrics import (
    aspect_coverage_score,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    source_attribution_coverage,
)
from src.evaluation.models import BenchmarkQuery, EvaluationReport, RetrievalMetrics
from src.evaluation.reporting import EvaluationReporter
from src.evaluation.reranker_eval import RerankerEvaluator
from src.evaluation.retrieval_eval import RetrievalEvaluator


# =====================================================================
# 1. Metric Math Unit Tests
# =====================================================================


def test_hit_rate_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    rel_ids = {"doc_c"}

    assert hit_rate_at_k(retrieved, rel_ids, k=1) == 0.0
    assert hit_rate_at_k(retrieved, rel_ids, k=2) == 0.0
    assert hit_rate_at_k(retrieved, rel_ids, k=3) == 1.0
    assert hit_rate_at_k(retrieved, rel_ids, k=5) == 1.0
    assert hit_rate_at_k(retrieved, rel_ids, k=0) == 0.0
    assert hit_rate_at_k(retrieved, set(), k=3) == 0.0


def test_recall_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    rel_ids = {"doc_b", "doc_d", "doc_z"}  # 3 relevant documents

    assert recall_at_k(retrieved, rel_ids, k=1) == 0.0
    assert recall_at_k(retrieved, rel_ids, k=2) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(retrieved, rel_ids, k=4) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(retrieved, set(), k=4) == 0.0


def test_precision_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c"]
    rel_ids = {"doc_b"}

    # k=1: 0/1 = 0.0
    assert precision_at_k(retrieved, rel_ids, k=1) == 0.0
    # k=2: 1/2 = 0.5
    assert precision_at_k(retrieved, rel_ids, k=2) == 0.5
    # k=5: 1/5 = 0.2 (strictly divided by K)
    assert precision_at_k(retrieved, rel_ids, k=5) == 0.2


def test_reciprocal_rank():
    retrieved = ["doc_x", "doc_y", "doc_target", "doc_z"]
    rel_ids = {"doc_target"}

    assert reciprocal_rank(retrieved, rel_ids, k=1) == 0.0
    assert reciprocal_rank(retrieved, rel_ids, k=2) == 0.0
    assert reciprocal_rank(retrieved, rel_ids, k=3) == pytest.approx(1.0 / 3.0)
    assert reciprocal_rank(retrieved, {"doc_missing"}, k=10) == 0.0


def test_ndcg_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    rel_ids = {"doc_b", "doc_d"}

    # At k=1: hits=0 -> 0.0
    assert ndcg_at_k(retrieved, rel_ids, k=1) == 0.0

    # At k=2: rank 2 has doc_b. DCG = 1/log2(3). IDCG (2 items at rank 1 & 2) = 1/log2(2) + 1/log2(3).
    dcg2 = 1.0 / math.log2(3)
    idcg2 = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert ndcg_at_k(retrieved, rel_ids, k=2) == pytest.approx(dcg2 / idcg2)

    # Edge cases
    assert ndcg_at_k(retrieved, set(), k=5) == 0.0
    assert ndcg_at_k([], rel_ids, k=5) == 0.0


def test_aspect_coverage_score():
    aspects = ["automatic DNS resolution", "better isolation", "environment variables"]
    text = "User-defined bridges offer automatic DNS resolution and provide better isolation between containers."

    # 2 out of 3 aspects present
    score = aspect_coverage_score(text, aspects)
    assert score == pytest.approx(2.0 / 3.0)

    # Empty aspects -> 1.0
    assert aspect_coverage_score(text, []) == 1.0
    # Empty text -> 0.0
    assert aspect_coverage_score("", aspects) == 0.0


def test_source_attribution_coverage():
    corpus_ids = {"chunk_1", "chunk_2", "chunk_3"}
    sources = [
        {"chunk_id": "chunk_1"},
        {"chunk_id": "chunk_2"},
        {"chunk_id": "chunk_fake_99"},
    ]

    cov = source_attribution_coverage(sources, corpus_ids)
    assert cov == pytest.approx(2.0 / 3.0)


# =====================================================================
# 2. Benchmark Dataset Integrity & Schema Tests
# =====================================================================


def test_benchmark_dataset_file_and_validation():
    dataset_path = Path("data/evaluation/benchmark_dataset.json")
    corpus_path = Path("data/processed/chunks.jsonl")

    assert dataset_path.exists(), "benchmark_dataset.json must exist"
    assert corpus_path.exists(), "chunks.jsonl must exist"

    queries, metadata = load_benchmark_dataset(dataset_path)
    assert len(queries) == 18, f"Expected 18 queries, got {len(queries)}"
    assert metadata["total_corpus_chunks"] == 308

    # Validate against actual corpus
    is_valid, errors = validate_benchmark_dataset(queries, corpus_path)
    assert is_valid, f"Dataset validation failed: {errors}"
    assert len(errors) == 0


# =====================================================================
# 3. Retrieval Evaluator Tests
# =====================================================================


def test_retrieval_evaluator_aggregation():
    q1 = BenchmarkQuery(
        id="q1",
        query="Test query 1",
        category="factual",
        relevant_chunk_ids=["chunk_1"],
        expected_refusal=False,
    )
    q2 = BenchmarkQuery(
        id="q2",
        query="Test query 2",
        category="factual",
        relevant_chunk_ids=["chunk_2"],
        expected_refusal=False,
    )
    # Refusal query (should be excluded from retrieval means)
    q3 = BenchmarkQuery(
        id="q3",
        query="Test refusal query",
        category="negative_ood",
        relevant_chunk_ids=[],
        expected_refusal=True,
    )

    retrieved_map = {
        "q1": ["chunk_1", "chunk_other"],  # Hit at rank 1
        "q2": ["chunk_other", "chunk_2"],  # Hit at rank 2
        "q3": ["chunk_other"],
    }

    evaluator = RetrievalEvaluator(
        dense_retriever=MagicMock(),
        sparse_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        reranker=MagicMock(),
    )

    metrics = evaluator.evaluate_retriever(
        queries=[q1, q2, q3],
        pipeline_name="mock_pipeline",
        retrieved_ids_map=retrieved_map,
        k_values=[1, 3],
    )

    # q1: Hit@1=1.0, MRR=1.0. q2: Hit@1=0.0, MRR=0.5.
    # Mean Hit@1 = (1.0 + 0.0)/2 = 0.5. Mean MRR = (1.0 + 0.5)/2 = 0.75.
    assert metrics.hit_rate[1] == 0.5
    assert metrics.hit_rate[3] == 1.0
    assert metrics.mrr == 0.75


# =====================================================================
# 4. Reranker Evaluator Tests
# =====================================================================


def test_reranker_evaluator_promotion():
    q1 = BenchmarkQuery(
        id="q1",
        query="Rerank test",
        category="factual",
        relevant_chunk_ids=["chunk_target"],
        expected_refusal=False,
    )

    pre_map = {"q1": ["chunk_a", "chunk_b", "chunk_c", "chunk_target"]}  # Rank 4
    post_map = {"q1": ["chunk_target", "chunk_a", "chunk_b", "chunk_c"]}  # Rank 1

    evaluator = RerankerEvaluator()
    metrics = evaluator.evaluate_reranker(
        queries=[q1],
        pre_rerank_map=pre_map,
        post_rerank_map=post_map,
    )

    assert metrics.pre_rerank_mrr == 0.25
    assert metrics.post_rerank_mrr == 1.0
    assert metrics.delta_mrr == 0.75
    assert metrics.pre_rerank_hit_1 == 0.0
    assert metrics.post_rerank_hit_1 == 1.0
    assert metrics.delta_hit_1 == 1.0
    assert metrics.promotion_rate == 1.0
    assert metrics.average_rank_shift == 3.0  # moved from 4 to 1


# =====================================================================
# 5. CRAG & Answer Evaluator Tests
# =====================================================================


def test_crag_evaluator_transitions():
    q1 = BenchmarkQuery(id="q1", query="Query 1", expected_refusal=False, relevant_chunk_ids=["c1"])
    q2 = BenchmarkQuery(id="q2", query="Query 2", expected_refusal=True, relevant_chunk_ids=[])

    resp1 = AgentResponse(
        query="Query 1",
        answer="Valid answer",
        hops_executed=2,
        is_corrected=True,
        final_evidence_grade=EvidenceGrade.GOOD,
        correction_traces=[
            CorrectionTrace(hop_index=1, evidence_grade="PARTIAL", query_used="Query 1"),
            CorrectionTrace(hop_index=2, evidence_grade="GOOD", query_used="Query 1 targeted"),
        ],
    )
    resp2 = AgentResponse(
        query="Query 2",
        answer="Based on the available local technical documentation, there is insufficient evidence to answer this question.",
        hops_executed=1,
        is_corrected=False,
        final_evidence_grade=EvidenceGrade.BAD,
    )

    evaluator = CRAGEvaluator()
    metrics = evaluator.evaluate_crag([q1, q2], [resp1, resp2])

    assert metrics.total_queries == 2
    assert metrics.correction_trigger_count == 1
    assert metrics.correction_trigger_rate == 0.5
    assert metrics.successful_correction_count == 1
    assert metrics.successful_correction_rate == 1.0
    assert metrics.refusal_accuracy == 1.0
    assert metrics.average_hops == 1.5


def test_answer_evaluator_aspects_and_refusal():
    q1 = BenchmarkQuery(
        id="q1",
        query="What is bridge?",
        expected_aspects=["isolation", "network"],
        expected_refusal=False,
    )
    resp1 = AgentResponse(
        query="What is bridge?",
        answer="A bridge provides isolation and custom network connectivity.",
        sources=[{"chunk_id": "c1"}],
    )

    evaluator = AnswerEvaluator(corpus_chunk_ids={"c1", "c2"})
    metrics = evaluator.evaluate_answers([q1], [resp1])

    assert metrics.mean_aspect_coverage == 1.0
    assert metrics.refusal_accuracy == 1.0
    assert metrics.mean_source_attribution_coverage == 1.0


# =====================================================================
# 6. Reporting & Serialization Tests
# =====================================================================


def test_evaluation_reporter_json_and_csv(tmp_path: Path):
    report = EvaluationReport(
        run_id="test_run_01",
        timestamp="2026-09-02T21:00:00Z",
        mode="fast",
        git_commit="abcdef",
        dataset_version="1.0.0",
        dataset_hash="hash123",
        execution_duration_sec=0.45,
        environment={"python": "3.11"},
        retrieval_comparison={"dense": RetrievalMetrics(hit_rate={1: 0.8}, mrr=0.85)},
    )

    json_file = EvaluationReporter.save_json(report, tmp_path)
    csv_file = EvaluationReporter.save_csv(report, tmp_path)

    assert json_file.exists()
    assert csv_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["run_id"] == "test_run_01"
        assert loaded["mode"] == "fast"
