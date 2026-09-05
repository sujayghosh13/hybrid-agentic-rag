import pytest
from src.agent.llm import MockOllamaClient
from src.correction.corrective_action import CorrectiveActionEngine
from src.correction.evaluator import EvidenceEvaluator
from src.correction.models import CorrectionTrace, EvidenceEvaluation, EvidenceGrade
from src.reranking.models import RerankedResult


def _make_dummy_chunk(chunk_id: str, text: str, score: float = 1.0) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id,
        text=text,
        source="docker-doc.html",
        metadata={"filename": "docker-doc.html"},
        score=score,
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.03,
        rerank_score=score,
        rerank_rank=1,
    )


def test_deterministic_bad_on_empty_chunks():
    evaluator = EvidenceEvaluator(llm_client=MockOllamaClient(default_response="GRADE: GOOD\nMISSING: NONE\nREASON: Good."))
    result = evaluator.evaluate("How does Docker bridge work?", [])
    assert result.grade == EvidenceGrade.BAD
    assert "No candidate chunks" in (result.reason or "")


def test_deterministic_bad_on_low_rerank_score():
    evaluator = EvidenceEvaluator(llm_client=MockOllamaClient(default_response="GRADE: GOOD\nMISSING: NONE\nREASON: Good."))
    chunk = _make_dummy_chunk("chunk_1", "Irrelevant text", score=-9.5)
    result = evaluator.evaluate("How does Docker bridge work?", [chunk])
    assert result.grade == EvidenceGrade.BAD
    assert "below relevance threshold" in (result.reason or "")


def test_evaluator_good_parsing():
    mock_llm = MockOllamaClient(
        default_response="GRADE: GOOD\nMISSING: NONE\nREASON: All bridge networking facts present."
    )
    evaluator = EvidenceEvaluator(llm_client=mock_llm)
    chunk = _make_dummy_chunk("chunk_1", "Docker bridge network driver configuration.", score=2.5)
    result = evaluator.evaluate("How does Docker bridge work?", [chunk])

    assert result.grade == EvidenceGrade.GOOD
    assert result.missing_aspect is None
    assert "All bridge networking facts present." in (result.reason or "")


def test_evaluator_partial_parsing_with_missing_aspect():
    mock_llm = MockOllamaClient(
        default_response="GRADE: PARTIAL\nMISSING: port publishing syntax and iptables\nREASON: Missing port details."
    )
    evaluator = EvidenceEvaluator(llm_client=mock_llm)
    chunk = _make_dummy_chunk("chunk_1", "Docker default bridge network.", score=1.8)
    result = evaluator.evaluate("How does Docker bridge port publishing work?", [chunk])

    assert result.grade == EvidenceGrade.PARTIAL
    assert result.missing_aspect == "port publishing syntax and iptables"
    assert "Missing port details." in (result.reason or "")


def test_evaluator_bad_parsing_with_reason():
    mock_llm = MockOllamaClient(
        default_response="GRADE: BAD\nMISSING: NONE\nREASON: Context discusses Kubernetes pods instead of Docker bridge."
    )
    evaluator = EvidenceEvaluator(llm_client=mock_llm)
    chunk = _make_dummy_chunk("chunk_1", "Kubernetes Pod lifecycle.", score=0.5)
    result = evaluator.evaluate("How does Docker bridge work?", [chunk])

    assert result.grade == EvidenceGrade.BAD
    assert "Kubernetes pods instead of Docker bridge" in (result.reason or "")


def test_evaluator_error_fallback():
    def raise_err(_):
        raise RuntimeError("Simulated connection error")

    mock_llm = MockOllamaClient(response_generator=raise_err)
    evaluator = EvidenceEvaluator(llm_client=mock_llm)
    chunk = _make_dummy_chunk("chunk_1", "Some technical context.", score=1.0)
    result = evaluator.evaluate("Any question", [chunk])

    # Should fall back safely to GOOD without crashing
    assert result.grade == EvidenceGrade.GOOD
    assert "Evaluator error fallback" in (result.reason or "")


def test_corrective_query_generation_good():
    engine = CorrectiveActionEngine(llm_client=MockOllamaClient(default_response="Something"))
    eval_good = EvidenceEvaluation(grade=EvidenceGrade.GOOD)
    query = engine.generate_corrective_query("How does bridge work?", eval_good)
    assert query == "How does bridge work?"


def test_corrective_query_generation_partial():
    mock_llm = MockOllamaClient(default_response="Docker bridge port publishing iptables")
    engine = CorrectiveActionEngine(llm_client=mock_llm)
    eval_partial = EvidenceEvaluation(
        grade=EvidenceGrade.PARTIAL,
        missing_aspect="port publishing iptables",
    )
    query = engine.generate_corrective_query("How does Docker bridge networking work?", eval_partial)
    assert query == "Docker bridge port publishing iptables"


def test_corrective_query_generation_bad():
    mock_llm = MockOllamaClient(default_response="Docker bridge network driver configuration")
    engine = CorrectiveActionEngine(llm_client=mock_llm)
    eval_bad = EvidenceEvaluation(
        grade=EvidenceGrade.BAD,
        reason="Irrelevant pods context",
    )
    query = engine.generate_corrective_query("How does Docker bridge networking work?", eval_bad)
    assert query == "Docker bridge network driver configuration"


def test_models_to_dict():
    eval_obj = EvidenceEvaluation(grade=EvidenceGrade.PARTIAL, missing_aspect="dns", reason="missing dns")
    d_eval = eval_obj.to_dict()
    assert d_eval["grade"] == "PARTIAL"
    assert d_eval["missing_aspect"] == "dns"

    trace = CorrectionTrace(
        hop_index=1,
        query_used="test query",
        evidence_grade=EvidenceGrade.PARTIAL,
        missing_aspect="dns",
        reason="partial evidence",
        action_taken="TRIGGER_CORRECTION",
        candidates_retrieved=20,
        reranked_chunks_added=5,
    )
    d_trace = trace.to_dict()
    assert d_trace["hop_index"] == 1
    assert d_trace["evidence_grade"] == "PARTIAL"
    assert d_trace["candidates_retrieved"] == 20


def test_evaluator_high_confidence_fast_path():
    """Verify that scores >= crag_high_confidence_score bypass the LLM grader and return GOOD directly."""
    # LLM should never be called; if called, raise an error
    def should_not_be_called(_):
        raise AssertionError("LLM grader was invoked despite high-confidence fast-path!")

    mock_llm = MockOllamaClient(response_generator=should_not_be_called)
    evaluator = EvidenceEvaluator(llm_client=mock_llm)

    # Chunk score = 4.2 >= crag_high_confidence_score (3.5)
    chunk = _make_dummy_chunk("chunk_1", "Comprehensive Docker bridge networking guide.", score=4.2)
    result = evaluator.evaluate("How does Docker bridge networking work?", [chunk])

    assert result.grade == EvidenceGrade.GOOD
    assert "exceeds high-confidence threshold" in (result.reason or "")
