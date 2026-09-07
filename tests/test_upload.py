import io
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from src.api.dependencies import get_rag_service
from src.api.main import app
from src.api.schemas import DocumentUploadResponse, ReadinessStatus
from src.api.service import RAGService
from src.ingestion.indexer import (
    DocumentIndexer,
    DocumentIndexerError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from src.ingestion.models import Chunk, ChunkMetadata
from src.ui.api_client import (
    APIServerError,
    APIValidationError,
    RAGApiClient,
)


# Minimal valid PDF binary stream containing extractable text
MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 84 >>
stream
BT
/F1 18 Tf
50 700 Td
(Kubernetes Pod Lifecycle and Autoscaling Guide) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000378 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
455
%%EOF
"""


class MockDenseRetriever:
    """Mock dense retriever to avoid loading heavy SentenceTransformer in unit tests."""

    def __init__(self):
        self.upserted_chunks = []

    def upsert_chunks(self, chunks, batch_size=64):
        self.upserted_chunks.extend(chunks)
        return len(chunks)


class MockSparseRetriever:
    """Mock sparse retriever for unit tests."""

    def __init__(self):
        self.indexed_chunks = []

    def index_chunks(self, chunks, save_path=None):
        self.indexed_chunks = list(chunks)
        return len(chunks)

    def reload_index(self, source_path=None):
        return True


@pytest.fixture
def temp_environment():
    """Create isolated temporary directories for indexing tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir) / "raw"
        raw_dir.mkdir()
        chunks_file = Path(tmp_dir) / "chunks.jsonl"
        bm25_file = Path(tmp_dir) / "bm25_index.pkl"

        indexer = DocumentIndexer(
            raw_dir=raw_dir,
            chunks_path=chunks_file,
            bm25_path=bm25_file,
            dense_retriever=MockDenseRetriever(),
            sparse_retriever=MockSparseRetriever(),
        )
        yield {
            "indexer": indexer,
            "raw_dir": raw_dir,
            "chunks_file": chunks_file,
            "bm25_file": bm25_file,
        }


# =====================================================================
# 1. DocumentIndexer Unit Tests
# =====================================================================


def test_indexer_markdown_success(temp_environment):
    """Indexing a valid markdown document generates chunks and updates indexes."""
    indexer = temp_environment["indexer"]
    md_content = b"# Docker Overlay Network\n\nOverlay networks connect multiple Docker daemons together."
    
    progress_steps = []
    result = indexer.index_uploaded_file(
        filename="overlay-network.md",
        content=md_content,
        on_progress=lambda step: progress_steps.append(step),
    )

    assert result["status"] == "success"
    assert result["filename"] == "overlay-network.md"
    assert result["chunks_indexed"] >= 1
    assert result["total_chunks"] >= 1
    assert "Upload received" in progress_steps
    assert "Extracting text..." in progress_steps
    assert "Creating chunks..." in progress_steps
    assert "Generating embeddings..." in progress_steps
    assert "Updating indexes..." in progress_steps
    assert "✅ Document indexed successfully" in progress_steps


def test_indexer_html_success(temp_environment):
    """Indexing a valid HTML document succeeds and attaches metadata."""
    indexer = temp_environment["indexer"]
    html_content = b"<html><body><h1>Kubernetes Pods</h1><p>A Pod is the basic execution unit in K8s.</p></body></html>"

    result = indexer.index_uploaded_file(
        filename="k8s-pods.html",
        content=html_content,
    )

    assert result["status"] == "success"
    assert result["filename"] == "k8s-pods.html"
    assert result["chunks_indexed"] >= 1


def test_indexer_pdf_success(temp_environment):
    """Indexing a valid PDF document succeeds."""
    indexer = temp_environment["indexer"]

    result = indexer.index_uploaded_file(
        filename="k8s-lifecycle.pdf",
        content=MINIMAL_PDF_BYTES,
    )

    assert result["status"] == "success"
    assert result["filename"] == "k8s-lifecycle.pdf"
    assert result["chunks_indexed"] >= 1


def test_indexer_duplicate_skip(temp_environment):
    """Re-uploading the exact same content skips re-indexing."""
    indexer = temp_environment["indexer"]
    content = b"# Ingress Controller\n\nIngress exposes HTTP and HTTPS routes from outside the cluster."

    # First upload
    res1 = indexer.index_uploaded_file("ingress.md", content)
    assert res1["status"] == "success"

    # Second upload with identical content
    res2 = indexer.index_uploaded_file("ingress.md", content)
    assert res2["status"] == "skipped"
    assert "already indexed with identical content" in res2["message"]


def test_indexer_unsupported_file_type(temp_environment):
    """Uploading an unsupported extension raises UnsupportedFileTypeError."""
    indexer = temp_environment["indexer"]

    with pytest.raises(UnsupportedFileTypeError) as exc:
        indexer.index_uploaded_file("archive.zip", b"PK000fakecontent")
    assert "Unsupported file format" in str(exc.value)


def test_indexer_empty_file(temp_environment):
    """Uploading empty bytes raises EmptyDocumentError."""
    indexer = temp_environment["indexer"]

    with pytest.raises(EmptyDocumentError) as exc:
        indexer.index_uploaded_file("empty.md", b"   ")
    assert "empty" in str(exc.value)


# =====================================================================
# 2. FastAPI Endpoint Tests (POST /documents/upload)
# =====================================================================


@pytest.fixture
def mock_upload_service():
    """Mock RAGService with upload_and_index."""
    service = MagicMock(spec=RAGService)
    return service


@pytest.fixture
def test_client(mock_upload_service):
    """TestClient with overridden RAGService."""
    app.dependency_overrides[get_rag_service] = lambda: mock_upload_service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_api_upload_success(test_client, mock_upload_service):
    """POST /documents/upload with valid file returns 200 and DocumentUploadResponse."""
    async def mock_upload(filename, content):
        return DocumentUploadResponse(
            status="success",
            filename=filename,
            chunks_indexed=2,
            total_chunks=310,
            message="Document indexed successfully.",
        )

    mock_upload_service.upload_and_index = mock_upload

    files = {"file": ("networking.md", b"# Networking\nDetails", "text/markdown")}
    response = test_client.post("/documents/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "networking.md"
    assert data["chunks_indexed"] == 2
    assert data["total_chunks"] == 310


def test_api_upload_unsupported_type(test_client, mock_upload_service):
    """POST /documents/upload with unsupported file returns 400."""
    async def mock_upload(filename, content):
        raise UnsupportedFileTypeError("Unsupported file format '.exe'")

    mock_upload_service.upload_and_index = mock_upload

    files = {"file": ("malware.exe", b"binary", "application/octet-stream")}
    response = test_client.post("/documents/upload", files=files)

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_api_upload_empty_document(test_client, mock_upload_service):
    """POST /documents/upload with empty document returns 400."""
    async def mock_upload(filename, content):
        raise EmptyDocumentError("Document is empty")

    mock_upload_service.upload_and_index = mock_upload

    files = {"file": ("empty.md", b"", "text/markdown")}
    response = test_client.post("/documents/upload", files=files)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


# =====================================================================
# 3. RAGApiClient Upload Tests
# =====================================================================


def test_client_upload_success():
    """RAGApiClient.upload_document sends POST and returns parsed response."""
    client = RAGApiClient(base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "filename": "k8s-service.md",
        "chunks_indexed": 3,
        "total_chunks": 311,
        "message": "Indexed",
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.upload_document("k8s-service.md", b"# Service\nSpec")
        assert res["status"] == "success"
        assert res["chunks_indexed"] == 3


def test_client_upload_validation_error():
    """RAGApiClient.upload_document raises APIValidationError on 400."""
    client = RAGApiClient(base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"detail": "Unsupported file format .xyz"}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(APIValidationError) as exc:
            client.upload_document("file.xyz", b"content")
        assert "Unsupported file format" in str(exc.value)


def test_client_upload_server_error():
    """RAGApiClient.upload_document raises APIServerError on 500."""
    client = RAGApiClient(base_url="http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"detail": "Qdrant connection failed"}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(APIServerError) as exc:
            client.upload_document("file.md", b"content")
        assert "Server error during indexing" in str(exc.value)
