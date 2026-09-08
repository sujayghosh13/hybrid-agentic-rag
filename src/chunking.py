"""Capstone Structural Compatibility Wrapper for Document Chunking.

Re-exports the structure-aware chunker and chunk data models from
`src.ingestion.chunker` to satisfy capstone specification Section 22.
"""

from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.models import Chunk, ChunkMetadata

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "StructureAwareChunker",
]
