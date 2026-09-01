import atexit
import logging
from pathlib import Path
import sys
import uuid
from typing import Any, List, Optional

# Preload msvcrt on Windows to prevent late-import during interpreter teardown
if sys.platform == "win32":
    try:
        import msvcrt
    except ImportError:
        pass

import numpy as np
from src.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from src.ingestion.models import Chunk
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense vector retriever using SentenceTransformers and Qdrant vector database."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        embedder: Optional[Any] = None,
    ):
        self.collection_name = collection_name or settings.qdrant_collection
        self.model_name = embedding_model_name or settings.embedding_model_name

        # Qdrant client connection (supports in-memory client or remote host)
        if client is not None:
            self.client = client
        else:
            try:
                logger.info(f"Connecting to Qdrant server at {settings.qdrant_url}...")
                self.client = QdrantClient(url=settings.qdrant_url, timeout=2.0)
                # Verify server connectivity
                self.client.get_collections()
            except Exception as e:
                logger.warning(
                    f"Could not connect to Qdrant server at '{settings.qdrant_url}' ({e}). "
                    f"Falling back to embedded local storage at 'data/processed/qdrant_storage'."
                )
                storage_path = Path("data/processed/qdrant_storage")
                storage_path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(storage_path))

        # Register graceful cleanup before interpreter module teardown
        atexit.register(self.close)

        # Sentence Transformer embedder instance
        if embedder is not None:
            self.embedder = embedder
        else:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)

        # Get embedding dimension dynamically
        dummy_vec = self.embedder.encode("test", convert_to_numpy=True)
        self.vector_size = int(np.asarray(dummy_vec).squeeze().shape[-1])

    def close(self) -> None:
        """Explicitly close the Qdrant client connection."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_collection(self, recreate: bool = True) -> None:
        """Create or recreate the Qdrant collection."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            logger.info(f"Recreating Qdrant collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (vector_size={self.vector_size})...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Embed and upsert list of Chunk objects into Qdrant."""
        if not chunks:
            return 0

        self.create_collection(recreate=True)

        points: List[qmodels.PointStruct] = []
        texts = [chunk.text for chunk in chunks]

        logger.info(f"Generating dense embeddings for {len(chunks)} chunks...")
        embeddings = self.embedder.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

        for chunk, embedding in zip(chunks, embeddings):
            # Deterministic UUID v5 derived from string chunk.id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload = {
                "chunk_id": chunk.id,
                "text": chunk.text,
                "source": chunk.source,
                "metadata": chunk.metadata.to_dict(),
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        logger.info(f"Upserting {len(points)} points into Qdrant collection '{self.collection_name}'...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return len(points)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform dense vector search for a given query."""
        if not query.strip():
            return []

        query_vector = self.embedder.encode(query, convert_to_numpy=True).tolist()

        # Handle qdrant-client versions API
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
        except AttributeError:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )
            hits = response.points

        results: List[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id", str(hit.id))
            text = payload.get("text", "")
            source = payload.get("source", "")
            metadata = payload.get("metadata", {})

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                    metadata=metadata,
                    score=float(hit.score),
                    dense_rank=rank,
                )
            )

        return results
