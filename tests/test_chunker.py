from pathlib import Path
import pytest

from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.models import DocType, DocumentElement, RawDocument


@pytest.fixture
def chunker():
    return StructureAwareChunker(min_tokens=30, max_tokens=60, overlap_ratio=0.20)


def test_short_document(chunker):
    """A document shorter than target chunk size should produce a single chunk."""
    short_text = "This is a short technical document with just a few words."
    doc = RawDocument(
        filepath=Path("test_short.md"),
        filename="test_short.md",
        doc_type=DocType.MARKDOWN,
        elements=[
            DocumentElement(
                text=short_text,
                heading="Overview",
                section_path=["Overview"],
            )
        ],
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == short_text
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[0].metadata.total_chunks == 1
    assert chunks[0].metadata.heading == "Overview"
    assert chunks[0].metadata.section == "Overview"


def test_normal_document_splitting(chunker):
    """A long document should be split into multiple chunks respecting token boundaries."""
    elements = []
    for i in range(20):
        elements.append(
            DocumentElement(
                text=f"Paragraph {i}: " + "Detailed technical information about system architecture. " * 3,
                heading="Architecture",
                section_path=["System", "Architecture"],
            )
        )

    doc = RawDocument(
        filepath=Path("architecture.md"),
        filename="architecture.md",
        doc_type=DocType.MARKDOWN,
        elements=elements,
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.token_count <= chunker.max_tokens + 20  # allow small margin for boundary text
        assert chunk.metadata.filename == "architecture.md"
        assert chunk.metadata.doc_type == "markdown"


def test_preservation_of_headings_sections(chunker):
    """Heading and section path context must be preserved in chunk metadata."""
    doc = RawDocument(
        filepath=Path("guide.md"),
        filename="guide.md",
        doc_type=DocType.MARKDOWN,
        elements=[
            DocumentElement(
                text="Content under section 1.",
                heading="Section 1",
                section_path=["Guide", "Section 1"],
            ),
            DocumentElement(
                text="Content under subsection 1.1.",
                heading="Subsection 1.1",
                section_path=["Guide", "Section 1", "Subsection 1.1"],
            ),
        ],
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    assert chunks[0].metadata.section == "Guide > Section 1"
    assert chunks[0].metadata.heading == "Section 1"


def test_overlap_behavior(chunker):
    """Adjacent chunks must share overlapping text context."""
    elements = [
        DocumentElement(
            text=f"Sentence block {i}: " + "Important context word sequence for testing overlap behavior. " * 2,
            heading="Overlap Test",
            section_path=["Test"],
        )
        for i in range(15)
    ]

    doc = RawDocument(
        filepath=Path("overlap.md"),
        filename="overlap.md",
        doc_type=DocType.MARKDOWN,
        elements=elements,
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        chunk1_text = chunks[i].text
        chunk2_text = chunks[i + 1].text

        # Get last few words of chunk 1
        chunk1_tail_words = set(chunk1_text.split()[-8:])
        chunk2_head_words = set(chunk2_text.split()[:15])

        # Verify non-empty intersection demonstrating overlap
        common_words = chunk1_tail_words.intersection(chunk2_head_words)
        assert len(common_words) > 0, f"No overlap found between chunk {i} and chunk {i+1}"


def test_metadata_generation(chunker):
    """Verify all required metadata fields are properly populated."""
    doc = RawDocument(
        filepath=Path("doc.pdf"),
        filename="doc.pdf",
        doc_type=DocType.PDF,
        elements=[
            DocumentElement(
                text="Page 3 technical document text content.",
                heading="PDF Page 3",
                section_path=["PDF Page 3"],
                page_number=3,
            )
        ],
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta.filename == "doc.pdf"
    assert meta.doc_type == "pdf"
    assert meta.page_number == 3
    assert meta.section == "PDF Page 3"
    assert meta.heading == "PDF Page 3"
    assert meta.chunk_index == 0
    assert meta.total_chunks == 1
    assert meta.token_count > 0
    assert meta.char_count > 0

    chunk_dict = chunks[0].to_dict()
    assert "id" in chunk_dict
    assert "text" in chunk_dict
    assert "source" in chunk_dict
    assert "metadata" in chunk_dict
