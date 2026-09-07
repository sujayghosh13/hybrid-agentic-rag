import pytest
from pathlib import Path

from src.ingestion.models import Chunk, ChunkMetadata
from src.retrieval.models import SearchResult
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.agent.tools import HybridSearchTool, RerankTool
from src.agent.agent import LocalQwenAgent
from src.agent.llm import MockOllamaClient
from src.reranking.cross_encoder import CrossEncoderReranker


def _create_sample_chunks():
    c1 = Chunk(
        id="doc1_c1",
        text="Docker bridge network configuration and host routing details.",
        source="docker-bridge.html",
        metadata=ChunkMetadata(
            filename="docker-bridge.html",
            doc_type="html",
            section="Bridge Driver",
            heading="Configuration",
            chunk_index=0,
            total_chunks=1,
        ),
    )
    c2 = Chunk(
        id="doc2_c1",
        text="Kubernetes Pod disruption budgets and minAvailable replica policies.",
        source="kubernetes-pdb.pdf",
        metadata=ChunkMetadata(
            filename="kubernetes-pdb.pdf",
            doc_type="pdf",
            section="PDB Policy",
            heading="Disruption",
            chunk_index=0,
            total_chunks=1,
        ),
    )
    c3 = Chunk(
        id="doc3_c1",
        text="Kubernetes Pod lifecycle, init containers, and restart policies in markdown format.",
        source="kubernetes-pods.md",
        metadata=ChunkMetadata(
            filename="kubernetes-pods.md",
            doc_type="markdown",
            section="Pod Lifecycle",
            heading="Lifecycle",
            chunk_index=0,
            total_chunks=1,
        ),
    )
    c4 = Chunk(
        id="doc4_c1",
        text="Enterprise cloud deployment architectures documented in Microsoft Word format.",
        source="architecture.docx",
        metadata=ChunkMetadata(
            filename="architecture.docx",
            doc_type="docx",
            section="Architecture",
            heading="Cloud",
            chunk_index=0,
            total_chunks=1,
        ),
    )
    # Background chunks so common terms in small corpus have positive IDF
    c5 = Chunk(id="bg1", text="Linux kernel cgroups memory allocation.", source="linux.html", metadata=ChunkMetadata(filename="linux.html", doc_type="html", section="Kernel", heading="Memory", chunk_index=0, total_chunks=1))
    c6 = Chunk(id="bg2", text="Storage volume provisioners CSI driver plugins.", source="storage.html", metadata=ChunkMetadata(filename="storage.html", doc_type="html", section="Storage", heading="CSI", chunk_index=0, total_chunks=1))
    c7 = Chunk(id="bg3", text="Prometheus metrics scraping and Grafana dashboard alerts.", source="monitoring.html", metadata=ChunkMetadata(filename="monitoring.html", doc_type="html", section="Metrics", heading="Alerts", chunk_index=0, total_chunks=1))
    c8 = Chunk(id="bg4", text="TLS certificates and ingress gateway reverse proxy routing.", source="security.html", metadata=ChunkMetadata(filename="security.html", doc_type="html", section="Security", heading="TLS", chunk_index=0, total_chunks=1))
    return [c1, c2, c3, c4, c5, c6, c7, c8]


def test_bm25_metadata_filtering(tmp_path):
    chunks = _create_sample_chunks()
    bm25_file = tmp_path / "bm25.pkl"
    retriever = BM25Retriever(index_path=bm25_file)
    retriever.index_chunks(chunks, save_path=bm25_file)

    # 1. Unfiltered search returns all relevant hits
    unfiltered = retriever.search("Kubernetes Pod", top_k=10)
    assert len(unfiltered) >= 2

    # 2. Filter by doc_type="pdf" -> only PDF chunk should be returned
    pdf_only = retriever.search("Kubernetes Pod", top_k=10, filters={"doc_type": "pdf"})
    assert len(pdf_only) == 1
    assert pdf_only[0].metadata.get("doc_type") == "pdf"
    assert pdf_only[0].metadata.get("filename") == "kubernetes-pdb.pdf"

    # 3. Filter by filename="kubernetes-pods.md"
    md_only = retriever.search("Kubernetes Pod", top_k=10, filters={"filename": "kubernetes-pods.md"})
    assert len(md_only) == 1
    assert md_only[0].metadata.get("filename") == "kubernetes-pods.md"

    # 4. Filter by doc_type="docx"
    docx_only = retriever.search("deployment architectures", top_k=10, filters={"doc_type": "docx"})
    assert len(docx_only) == 1
    assert docx_only[0].metadata.get("doc_type") == "docx"

    # 5. Non-matching filter returns empty list
    none_match = retriever.search("Kubernetes Pod", top_k=10, filters={"filename": "non_existent.html"})
    assert len(none_match) == 0


def test_hybrid_search_with_filtering(tmp_path):
    chunks = _create_sample_chunks()
    bm25_file = tmp_path / "bm25.pkl"
    sparse_retriever = BM25Retriever(index_path=bm25_file)
    sparse_retriever.index_chunks(chunks, save_path=bm25_file)

    # Mock dense retriever that returns all chunks converted to SearchResult
    class MockDenseRetriever:
        def search(self, query: str, top_k: int = 10, filters=None):
            results = []
            for rank, c in enumerate(chunks, 1):
                meta = c.metadata.to_dict()
                if filters:
                    match = True
                    for fk, fv in filters.items():
                        if meta.get(fk) != fv:
                            match = False
                            break
                    if not match:
                        continue
                results.append(
                    SearchResult(
                        chunk_id=c.id,
                        text=c.text,
                        source=c.source,
                        metadata=meta,
                        score=1.0 - (rank * 0.1),
                        dense_rank=rank,
                    )
                )
            return results[:top_k]

    hybrid = HybridRetriever(
        dense_retriever=MockDenseRetriever(),
        sparse_retriever=sparse_retriever,
    )

    # Query with pdf filter
    results = hybrid.hybrid_search("Kubernetes", top_k=5, filters={"doc_type": "pdf"})
    assert len(results) >= 1
    assert all(r.metadata.get("doc_type") == "pdf" for r in results)

    # Query with docx filter
    docx_results = hybrid.hybrid_search("architecture", top_k=5, filters={"doc_type": "docx"})
    assert len(docx_results) == 1
    assert docx_results[0].metadata.get("doc_type") == "docx"


def test_agent_run_with_metadata_filters(tmp_path):
    chunks = _create_sample_chunks()
    bm25_file = tmp_path / "bm25.pkl"
    sparse_retriever = BM25Retriever(index_path=bm25_file)
    sparse_retriever.index_chunks(chunks, save_path=bm25_file)

    class FilterMockDense:
        def search(self, query: str, top_k: int = 10, filters=None):
            out = []
            for r, c in enumerate(chunks, 1):
                meta = c.metadata.to_dict()
                if filters:
                    match = True
                    for fk, fv in filters.items():
                        if meta.get(fk) != fv:
                            match = False
                            break
                    if not match:
                        continue
                out.append(SearchResult(chunk_id=c.id, text=c.text, source=c.source, metadata=meta, score=0.9, dense_rank=r))
            return out[:top_k]

    class PassThroughReranker:
        def rerank(self, query: str, candidates, top_k: int = 5):
            from src.reranking.models import RerankedResult
            return [
                RerankedResult(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    metadata=c.metadata,
                    score=c.score,
                    dense_rank=1,
                    sparse_rank=1,
                    rrf_score=0.05,
                    rerank_score=2.0,
                    rerank_rank=idx,
                )
                for idx, c in enumerate(candidates, 1)
            ][:top_k]

    hybrid = HybridRetriever(dense_retriever=FilterMockDense(), sparse_retriever=sparse_retriever)
    search_tool = HybridSearchTool(retriever=hybrid)
    rerank_tool = RerankTool(reranker=PassThroughReranker())

    agent = LocalQwenAgent(
        llm_client=MockOllamaClient(default_response="Filtered answer based only on PDF."),
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )

    response = agent.run("What is Kubernetes?", filters={"doc_type": "pdf"})
    assert response.retrieval_needed is True
    assert len(response.sources) > 0
    # Every returned source must match the filter
    for s in response.sources:
        assert s.get("metadata", {}).get("doc_type") == "pdf"
