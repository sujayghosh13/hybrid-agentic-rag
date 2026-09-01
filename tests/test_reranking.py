import pytest

from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.models import RerankedResult
from src.retrieval.models import SearchResult


class MockCrossEncoder:
    """Mock CrossEncoder model that returns deterministic scores based on query and document text."""

    def __init__(self, score_mapping=None):
        self.score_mapping = score_mapping or {}
        self.call_count = 0
        self.last_pairs = []

    def predict(self, pairs):
        self.call_count += 1
        self.last_pairs = pairs
        scores = []
        for query, doc in pairs:
            # Check explicit score map or use length-based heuristic
            if doc in self.score_mapping:
                scores.append(self.score_mapping[doc])
            elif "bridge" in doc.lower():
                scores.append(5.8)
            elif "networking" in doc.lower():
                scores.append(3.2)
            else:
                scores.append(0.5)
        return scores


@pytest.fixture
def sample_candidates():
    return [
        SearchResult(
            chunk_id="chunk_1",
            text="Kubernetes pods run container workloads.",
            source="k8s.html",
            metadata={"filename": "k8s.html", "section": "Workloads"},
            score=0.015,
            dense_rank=3,
            sparse_rank=5,
            rrf_score=0.015,
        ),
        SearchResult(
            chunk_id="chunk_2",
            text="Docker bridge networking driver connects containers on the same host.",
            source="docker.html",
            metadata={"filename": "docker.html", "section": "Bridge Driver"},
            score=0.032,
            dense_rank=1,
            sparse_rank=1,
            rrf_score=0.032,
        ),
        SearchResult(
            chunk_id="chunk_3",
            text="General networking concepts and network topology.",
            source="net.html",
            metadata={"filename": "net.html", "section": "Networking"},
            score=0.020,
            dense_rank=2,
            sparse_rank=4,
            rrf_score=0.020,
        ),
    ]


def test_reranker_initialization():
    """Verify CrossEncoderReranker initializes with mock model."""
    mock_model = MockCrossEncoder()
    reranker = CrossEncoderReranker(model=mock_model)
    assert reranker.model == mock_model


def test_pair_construction_and_scoring(sample_candidates):
    """Verify query-document pairs are properly constructed and scored."""
    mock_model = MockCrossEncoder()
    reranker = CrossEncoderReranker(model=mock_model)

    query = "How does Docker bridge networking work?"
    results = reranker.rerank(query=query, candidates=sample_candidates, top_k=3)

    assert mock_model.call_count == 1
    assert len(mock_model.last_pairs) == 3
    assert mock_model.last_pairs[0] == (query, sample_candidates[0].text)

    assert len(results) == 3
    assert all(isinstance(r, RerankedResult) for r in results)


def test_sorting_and_top_k(sample_candidates):
    """Verify candidates are sorted descending by rerank_score and sliced to top_k."""
    mock_model = MockCrossEncoder(
        score_mapping={
            sample_candidates[0].text: 1.2,
            sample_candidates[1].text: 9.5,
            sample_candidates[2].text: 4.8,
        }
    )
    reranker = CrossEncoderReranker(model=mock_model)

    # Request top 2 out of 3
    results = reranker.rerank(query="Docker bridge", candidates=sample_candidates, top_k=2)

    assert len(results) == 2
    # Highest score (chunk_2 = 9.5) should be rank 1
    assert results[0].chunk_id == "chunk_2"
    assert results[0].rerank_score == 9.5
    assert results[0].rerank_rank == 1

    # Second highest score (chunk_3 = 4.8) should be rank 2
    assert results[1].chunk_id == "chunk_3"
    assert results[1].rerank_score == 4.8
    assert results[1].rerank_rank == 2


def test_metadata_preservation(sample_candidates):
    """Verify all Phase 2 metadata and scores are preserved alongside rerank_score."""
    mock_model = MockCrossEncoder()
    reranker = CrossEncoderReranker(model=mock_model)

    results = reranker.rerank(query="bridge", candidates=sample_candidates, top_k=3)

    for res in results:
        assert res.chunk_id in ["chunk_1", "chunk_2", "chunk_3"]
        assert res.source != ""
        assert "filename" in res.metadata
        assert res.dense_rank is not None
        assert res.sparse_rank is not None
        assert res.rrf_score is not None
        assert res.rerank_score is not None
        assert res.rerank_rank in [1, 2, 3]

        as_dict = res.to_dict()
        assert "rerank_score" in as_dict
        assert "dense_rank" in as_dict
        assert "sparse_rank" in as_dict
        assert "rrf_score" in as_dict


def test_empty_and_edge_cases(sample_candidates):
    """Verify empty candidates, blank query, and invalid top_k handling."""
    mock_model = MockCrossEncoder()
    reranker = CrossEncoderReranker(model=mock_model)

    # Empty candidate list
    assert reranker.rerank(query="test", candidates=[], top_k=5) == []

    # Empty / whitespace query
    assert reranker.rerank(query="", candidates=sample_candidates, top_k=5) == []
    assert reranker.rerank(query="   ", candidates=sample_candidates, top_k=5) == []

    # Invalid top_k <= 0
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reranker.rerank(query="test", candidates=sample_candidates, top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reranker.rerank(query="test", candidates=sample_candidates, top_k=-1)
