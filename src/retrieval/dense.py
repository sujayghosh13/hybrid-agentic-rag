import atexit
import logging
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional

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

_shared_local_client: Optional[QdrantClient] = None
_remote_qdrant_available: Optional[bool] = None


class DenseRetriever:
    """Dense vector retriever using SentenceTransformers and Qdrant vector database."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        embedder: Optional[Any] = None,
    ):
        global _shared_local_client, _remote_qdrant_available
        self.collection_name = collection_name or settings.qdrant_collection
        self.model_name = embedding_model_name or settings.embedding_model_name
        self._query_embedding_cache: Dict[str, List[float]] = {}

        # Qdrant client connection (supports in-memory client or remote host)
        if client is not None:
            self.client = client
        elif _remote_qdrant_available is False:
            # Fast-path: Remote Qdrant was already confirmed unreachable in this process
            storage_path = Path("data/processed/qdrant_storage")
            storage_path.mkdir(parents=True, exist_ok=True)
            if _shared_local_client is None:
                _shared_local_client = QdrantClient(path=str(storage_path))
            self.client = _shared_local_client
        else:
            try:
                logger.info(f"Connecting to Qdrant server at {settings.qdrant_url}...")
                # Use lightweight 1.0s timeout to avoid 10-second blocking on offline server
                self.client = QdrantClient(url=settings.qdrant_url, timeout=1.0)
                # Verify server connectivity
                self.client.get_collections()
                _remote_qdrant_available = True
            except Exception as e:
                _remote_qdrant_available = False
                logger.warning(
                    f"Could not connect to Qdrant server at '{settings.qdrant_url}' ({e}). "
                    f"Falling back to embedded local storage at 'data/processed/qdrant_storage'."
                )
                storage_path = Path("data/processed/qdrant_storage")
                storage_path.mkdir(parents=True, exist_ok=True)
                if _shared_local_client is None:
                    _shared_local_client = QdrantClient(path=str(storage_path))
                self.client = _shared_local_client

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

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 64, recreate: bool = True) -> int:
        """Embed and upsert list of Chunk objects into Qdrant."""
        if not chunks:
            return 0

        self.create_collection(recreate=recreate)

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

    def delete_by_filename(self, filename: str) -> None:
        """Delete points associated with a specific filename from Qdrant collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.filename",
                            match=qmodels.MatchValue(value=filename),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted existing points for filename '{filename}' from collection '{self.collection_name}'.")
        except Exception as e:
            logger.warning(f"Could not delete existing points for '{filename}' from Qdrant: {e}")

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Embed and upsert chunks incrementally without recreating or wiping existing collection."""
        return self.index_chunks(chunks, batch_size=batch_size, recreate=False)

    def _encode_query(self, query: str) -> List[float]:
        """Encode query text into a vector, leveraging an in-memory cache to skip redundant model passes."""
        clean_q = query.strip()
        if clean_q in self._query_embedding_cache:
            return self._query_embedding_cache[clean_q]

        try:
            raw = self.embedder.encode(
                clean_q, convert_to_numpy=True, show_progress_bar=False
            )
        except TypeError:
            raw = self.embedder.encode(clean_q)

        if hasattr(raw, "tolist"):
            vec = raw.tolist()
        elif isinstance(raw, list):
            vec = raw
        else:
            vec = list(raw)

        if len(self._query_embedding_cache) >= 512:
            self._query_embedding_cache.pop(next(iter(self._query_embedding_cache)))
        self._query_embedding_cache[clean_q] = vec
        return vec

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform dense vector search for a given query with optional metadata filtering."""
        if not query.strip():
            return []

        query_vector = self._encode_query(query)

        # Build Qdrant filter condition if filters provided
        query_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                if v is not None and v != "":
                    payload_key = f"metadata.{k}" if not k.startswith("metadata.") else k
                    conditions.append(
                        qmodels.FieldCondition(
                            key=payload_key,
                            match=qmodels.MatchValue(value=v),
                        )
                    )
            if conditions:
                query_filter = qmodels.Filter(must=conditions)

        # Handle qdrant-client versions API (v1.10+ uses query_points, earlier used search)
        try:
            if hasattr(self.client, "query_points"):
                kwargs = {
                    "collection_name": self.collection_name,
                    "query": query_vector,
                    "limit": top_k,
                }
                if query_filter is not None:
                    kwargs["query_filter"] = query_filter
                response = self.client.query_points(**kwargs)
                hits = response.points
            elif hasattr(self.client, "search"):
                kwargs = {
                    "collection_name": self.collection_name,
                    "query_vector": query_vector,
                    "limit": top_k,
                }
                if query_filter is not None:
                    kwargs["query_filter"] = query_filter
                hits = self.client.search(**kwargs)
            else:
                logger.warning("QdrantClient has neither 'query_points' nor 'search' method.")
                return []
        except Exception as e:
            if "404" in str(e):
                logger.warning(
                    f"Qdrant collection '{self.collection_name}' not found on server (404). "
                    "Returning empty dense results. Run 'python scripts/build_index.py' to populate Qdrant."
                )
                return []
            raise

        if filters and hits:
            filtered_hits = []
            for hit in hits:
                payload = hit.payload or {}
                meta = payload.get("metadata", {})
                match = True
                for fk, fv in filters.items():
                    if fv is not None and fv != "":
                        val = meta.get(fk) if isinstance(meta, dict) else getattr(meta, fk, None)
                        if val is None:
                            val = payload.get(fk)
                        if val != fv:
                            match = False
                            break
                if match:
                    filtered_hits.append(hit)
            hits = filtered_hits[:top_k]

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
