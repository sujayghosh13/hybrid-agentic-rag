import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.writer import JSONLWriter, load_chunks_from_jsonl
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)


class DocumentIndexerError(Exception):
    """Base exception for document indexing errors."""
    pass


class UnsupportedFileTypeError(DocumentIndexerError):
    """Raised when file extension is not supported by IngestionPipeline."""
    pass


class EmptyDocumentError(DocumentIndexerError):
    """Raised when uploaded file contains no extractable text or chunks."""
    pass


class DocumentIndexer:
    """Thin orchestrator connecting file upload to existing IngestionPipeline, Qdrant, and BM25."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        chunks_path: Optional[Path] = None,
        bm25_path: Optional[Path] = None,
        pipeline: Optional[IngestionPipeline] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
    ):
        self.raw_dir = Path(raw_dir or "data/raw")
        self.chunks_path = Path(chunks_path or "data/processed/chunks.jsonl")
        self.bm25_path = Path(bm25_path or settings.bm25_index_path)
        self.pipeline = pipeline or IngestionPipeline()
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever

    @property
    def dense_retriever(self) -> DenseRetriever:
        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever()
        return self._dense_retriever

    @property
    def sparse_retriever(self) -> BM25Retriever:
        if self._sparse_retriever is None:
            self._sparse_retriever = BM25Retriever(index_path=self.bm25_path)
        return self._sparse_retriever

    def index_uploaded_file(
        self,
        filename: str,
        content: bytes,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Orchestrate incremental upload, chunking, embedding, and indexing."""
        def notify(msg: str):
            logger.info(f"[DocumentIndexer] {msg}")
            if on_progress:
                on_progress(msg)

        # 1. Upload received & validation
        notify("Upload received")
        clean_filename = Path(filename).name
        suffix = Path(clean_filename).suffix.lower()

        if suffix not in IngestionPipeline.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(IngestionPipeline.SUPPORTED_EXTENSIONS))
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{suffix}'. Supported formats: {supported}"
            )

        if not content or len(content.strip()) == 0:
            raise EmptyDocumentError(f"Uploaded file '{clean_filename}' is empty.")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        dest_file = self.raw_dir / clean_filename

        # Duplicate check: compare content hash against existing file and chunks
        content_hash = hashlib.sha256(content).hexdigest()
        existing_chunks = load_chunks_from_jsonl(self.chunks_path)
        has_existing_chunks = any(c.metadata.filename == clean_filename for c in existing_chunks)

        if dest_file.exists() and has_existing_chunks:
            try:
                existing_hash = hashlib.sha256(dest_file.read_bytes()).hexdigest()
                if existing_hash == content_hash:
                    notify("✅ Document already indexed (duplicate detected)")
                    chunk_count = sum(1 for c in existing_chunks if c.metadata.filename == clean_filename)
                    return {
                        "status": "skipped",
                        "filename": clean_filename,
                        "chunks_indexed": chunk_count,
                        "total_chunks": len(existing_chunks),
                        "message": f"Document '{clean_filename}' is already indexed with identical content. Skipped re-indexing.",
                    }
            except Exception as e:
                logger.warning(f"Could not verify existing file hash for '{clean_filename}': {e}")

        # If updating existing file with new content, purge old vectors from Qdrant
        if dest_file.exists() and has_existing_chunks:
            try:
                self.dense_retriever.delete_by_filename(clean_filename)
            except Exception as e:
                logger.warning(f"Could not purge old vector points for '{clean_filename}': {e}")

        # Write uploaded file to data/raw/
        dest_file.write_bytes(content)

        # 2. Extracting text...
        notify("Extracting text...")
        # 3. Creating chunks...
        notify("Creating chunks...")
        try:
            new_chunks = self.pipeline.process_file(dest_file)
        except Exception as e:
            logger.error(f"Failed to process file '{dest_file}': {e}", exc_info=True)
            raise DocumentIndexerError(f"Failed to process document '{clean_filename}': {e}") from e

        if not new_chunks:
            raise EmptyDocumentError(f"No text chunks could be extracted from '{clean_filename}'.")

        # 4. Generating embeddings...
        notify("Generating embeddings...")
        try:
            self.dense_retriever.upsert_chunks(new_chunks)
        except Exception as e:
            logger.error(f"Failed to generate embeddings / upsert to Qdrant: {e}", exc_info=True)
            raise DocumentIndexerError(f"Vector indexing failed for '{clean_filename}': {e}") from e

        # 5. Updating indexes...
        notify("Updating indexes...")
        # Merge chunks: filter out old chunks with this filename, append new chunks
        filtered_chunks = [c for c in existing_chunks if c.metadata.filename != clean_filename]
        all_chunks = filtered_chunks + new_chunks

        # Update chunks.jsonl
        JSONLWriter(self.chunks_path).write_chunks(all_chunks, append=False)

        # Re-index BM25 with updated chunk corpus
        self.sparse_retriever.index_chunks(all_chunks, save_path=self.bm25_path)

        # 6. Success
        notify("✅ Document indexed successfully")
        return {
            "status": "success",
            "filename": clean_filename,
            "chunks_indexed": len(new_chunks),
            "total_chunks": len(all_chunks),
            "message": f"Document '{clean_filename}' indexed successfully ({len(new_chunks)} chunks created, {len(all_chunks)} total in corpus).",
        }
