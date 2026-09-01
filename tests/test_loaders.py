from pathlib import Path
import tempfile
import pytest

from src.ingestion.loaders.factory import get_loader
from src.ingestion.loaders.html_loader import HTMLLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.models import DocType


def test_markdown_loader_headers():
    loader = MarkdownLoader()
    content = """# Title
Introductory text.

## Section 1
Details in section 1.

### Subsection 1.1
Deep details.
"""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        doc = loader.load(temp_path)
        assert doc.doc_type == DocType.MARKDOWN
        assert len(doc.elements) == 3

        assert doc.elements[0].heading == "Title"
        assert doc.elements[0].section_path == ["Title"]

        assert doc.elements[1].heading == "Section 1"
        assert doc.elements[1].section_path == ["Title", "Section 1"]

        assert doc.elements[2].heading == "Subsection 1.1"
        assert doc.elements[2].section_path == ["Title", "Section 1", "Subsection 1.1"]
    finally:
        temp_path.unlink(missing_ok=True)


def test_html_loader():
    loader = HTMLLoader()
    html_content = """<!DOCTYPE html>
<html>
<head><title>Test HTML</title></head>
<body>
  <h1>Main Heading</h1>
  <p>First paragraph text.</p>
  <h2>Sub Heading</h2>
  <p>Second paragraph text.</p>
</body>
</html>
"""
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False, encoding="utf-8") as f:
        f.write(html_content)
        temp_path = Path(f.name)

    try:
        doc = loader.load(temp_path)
        assert doc.doc_type == DocType.HTML
        assert len(doc.elements) >= 2
        assert doc.elements[0].heading == "Main Heading"
        assert "First paragraph text." in doc.elements[0].text
    finally:
        temp_path.unlink(missing_ok=True)


def test_factory_dispatch():
    md_loader = get_loader(Path("test.md"))
    assert isinstance(md_loader, MarkdownLoader)

    html_loader = get_loader(Path("test.html"))
    assert isinstance(html_loader, HTMLLoader)

    pdf_loader = get_loader(Path("test.pdf"))
    assert isinstance(pdf_loader, PDFLoader)

    with pytest.raises(ValueError):
        get_loader(Path("test.unknown"))
