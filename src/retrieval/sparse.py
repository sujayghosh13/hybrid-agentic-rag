import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.models import Chunk
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Sparse keyword retriever using BM25Okapi algorithm."""

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = Path(index_path or settings.bm25_index_path)
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.chunks_by_id: dict = {}

    def index_chunks(self, chunks: List[Chunk], save_path: Optional[Path] = None) -> int:
        """Build BM25 index over chunks and save to disk."""
        if not chunks:
            return 0

        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        corpus_tokens = [tokenize_text(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

        self.chunk_ids = [chunk.id for chunk in chunks]
        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}

        target_path = Path(save_path or self.index_path)
        self.save_index(target_path)
        return len(chunks)

    def save_index(self, target_path: Optional[Path] = None) -> None:
        """Save serialized BM25 index and chunk references to disk."""
        path = Path(target_path or self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "bm25": self.bm25,
            "chunk_ids": self.chunk_ids,
            "chunks_by_id": {cid: chunk.to_dict() for cid, chunk in self.chunks_by_id.items()},
        }

        logger.info(f"Saving BM25 index to '{path}'...")
        with path.open("wb") as f:
            pickle.dump(data, f)

    def load_index(self, source_path: Optional[Path] = None) -> bool:
        """Load serialized BM25 index from disk. Returns True if successful."""
        path = Path(source_path or self.index_path)
        if not path.exists():
            logger.warning(f"BM25 index file not found at '{path}'")
            return False

        logger.info(f"Loading BM25 index from '{path}'...")
        with path.open("rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.chunks_by_id = data["chunks_by_id"]
        return True

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform BM25 sparse keyword search for a given query."""
        if self.bm25 is None:
            if not self.load_index():
                raise RuntimeError("BM25 index is not loaded or built.")

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair index, score
        scored_indices = [(i, float(score)) for i, score in enumerate(scores) if score > 0]
        # Sort descending by score
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        top_hits = scored_indices[:top_k]

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(top_hits, start=1):
            chunk_id = self.chunk_ids[idx]
            chunk_data = self.chunks_by_id.get(chunk_id, {})

            if isinstance(chunk_data, dict):
                text = chunk_data.get("text", "")
                source = chunk_data.get("source", "")
                metadata = chunk_data.get("metadata", {})
            else:
                text = chunk_data.text
                source = chunk_data.source
                metadata = chunk_data.metadata.to_dict() if hasattr(chunk_data.metadata, "to_dict") else chunk_data.metadata

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                    metadata=metadata,
                    score=score,
                    sparse_rank=rank,
                )
            )

        return results
