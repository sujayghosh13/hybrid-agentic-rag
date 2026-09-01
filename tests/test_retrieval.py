from pathlib import Path
import tempfile
import pytest
from qdrant_client import QdrantClient

from src.ingestion.models import Chunk, ChunkMetadata
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import SearchResult
from src.retrieval.sparse import BM25Retriever


class DummyEmbedder:
    """Fast, deterministic dummy embedder for unit tests."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]

        import numpy as np

        results = []
        for sentence in sentences:
            # Simple deterministic hash-based vector
            h = hash(sentence) % 1000
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = (h / 1000.0)
            vec[1] = 1.0 - vec[0]
            results.append(vec)

        if is_single:
            return results[0]
        return np.array(results)


@pytest.fixture
def sample_chunks():
    chunks = []
    topics = [
        ("docker-networking", "Docker container networking allows isolated containers to communicate over bridge networks."),
        ("docker-bridge", "A bridge network uses a software bridge which lets containers connected to the same bridge network communicate."),
        ("k8s-pods", "Kubernetes Pods are the smallest deployable units of computing that you can create and manage in Kubernetes."),
        ("k8s-deployments", "A Deployment provides declarative updates for Pods and ReplicaSets in Kubernetes clusters."),
    ]
    for idx, (title, content) in enumerate(topics):
        meta = ChunkMetadata(
            filename=f"{title}.html",
            doc_type="html",
            heading=title.replace("-", " ").title(),
            section=title,
            chunk_index=idx,
            total_chunks=len(topics),
            token_count=len(content.split()),
            char_count=len(content),
        )
        chunk = Chunk(
            id=f"chunk_{idx}",
            text=content,
            source=f"data/raw/{title}.html",
            metadata=meta,
        )
        chunks.append(chunk)

    return chunks


def test_bm25_indexing_and_search(sample_chunks):
    """Test BM25 indexing and keyword search retrieval."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_file = Path(tmp_dir) / "bm25_test.pkl"
        retriever = BM25Retriever(index_path=index_file)

        indexed_count = retriever.index_chunks(sample_chunks, save_path=index_file)
        assert indexed_count == len(sample_chunks)
        assert index_file.exists()

        # Perform search
        results = retriever.search("bridge network", top_k=2)
        assert len(results) > 0
        assert results[0].sparse_rank == 1
        assert "chunk_" in results[0].chunk_id
        assert "metadata" in results[0].to_dict()


def test_qdrant_in_memory_dense_search(sample_chunks):
    """Test DenseRetriever using Qdrant in-memory client without requiring external Docker server."""
    in_memory_client = QdrantClient(location=":memory:")
    dummy_embedder = DummyEmbedder(dim=16)

    retriever = DenseRetriever(
        collection_name="test_collection",
        client=in_memory_client,
        embedder=dummy_embedder,
    )

    indexed_count = retriever.index_chunks(sample_chunks)
    assert indexed_count == len(sample_chunks)

    results = retriever.search("Kubernetes Pods", top_k=3)
    assert len(results) == 3
    assert results[0].dense_rank == 1
    assert results[0].chunk_id.startswith("chunk_")
    assert results[0].metadata["filename"] != ""


def test_reciprocal_rank_fusion_logic():
    """Test RRF algorithm scoring, ranking, and deduplication."""
    dense_res = [
        SearchResult(chunk_id="chunk_A", text="Text A", source="docA.md", score=0.9, dense_rank=1),
        SearchResult(chunk_id="chunk_B", text="Text B", source="docB.md", score=0.8, dense_rank=2),
    ]
    sparse_res = [
        SearchResult(chunk_id="chunk_B", text="Text B", source="docB.md", score=5.2, sparse_rank=1),
        SearchResult(chunk_id="chunk_C", text="Text C", source="docC.md", score=3.1, sparse_rank=2),
    ]

    fused = reciprocal_rank_fusion(dense_res, sparse_res, top_k=5, rrf_k=60)

    # Chunk B appears in both lists (dense_rank=2, sparse_rank=1)
    # RRF score B = 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
    # RRF score A = 1/(60+1) = 0.016393
    # RRF score C = 1/(60+2) = 0.016129
    assert len(fused) == 3
    assert fused[0].chunk_id == "chunk_B"  # Chunk B should rank first
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1

    # Verify deduplication
    chunk_ids = [res.chunk_id for res in fused]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_hybrid_retriever_end_to_end(sample_chunks):
    """Test full HybridRetriever with mock dense and BM25 sub-retrievers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bm25_path = Path(tmp_dir) / "bm25.pkl"
        sparse_retriever = BM25Retriever(index_path=bm25_path)
        sparse_retriever.index_chunks(sample_chunks, save_path=bm25_path)

        in_memory_client = QdrantClient(location=":memory:")
        dense_retriever = DenseRetriever(
            collection_name="test_hybrid",
            client=in_memory_client,
            embedder=DummyEmbedder(dim=16),
        )
        dense_retriever.index_chunks(sample_chunks)

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            rrf_k=60,
        )

        results = hybrid_retriever.hybrid_search("container networking", top_k=2)

        assert len(results) == 2
        for res in results:
            assert res.chunk_id in [c.id for c in sample_chunks]
            assert res.rrf_score is not None
            assert res.text != ""
            assert "filename" in res.metadata
