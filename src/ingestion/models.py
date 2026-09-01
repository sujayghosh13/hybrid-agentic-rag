from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "DocType":
        normalized = ext.lower().lstrip(".")
        if normalized in ("md", "markdown"):
            return cls.MARKDOWN
        elif normalized in ("html", "htm"):
            return cls.HTML
        elif normalized == "pdf":
            return cls.PDF
        return cls.UNKNOWN


@dataclass
class DocumentElement:
    """Represents a structural element within a parsed document."""
    text: str
    heading: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    page_number: Optional[int] = None

    @property
    def section_string(self) -> Optional[str]:
        """Format section path hierarchy as a string (e.g. 'Intro > Overview')."""
        if not self.section_path:
            return None
        return " > ".join(self.section_path)


@dataclass
class RawDocument:
    """Represents a raw document loaded into memory prior to chunking."""
    filepath: Path
    filename: str
    doc_type: DocType
    elements: List[DocumentElement] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """Metadata attached to an individual document chunk."""
    filename: str
    doc_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    char_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "filename": self.filename,
            "doc_type": self.doc_type,
            "page_number": self.page_number,
            "section": self.section,
            "heading": self.heading,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "char_count": self.char_count,
        }


@dataclass
class Chunk:
    """Output unit for JSONL export."""
    id: str
    text: str
    source: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to a JSONL-compatible dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
        }
