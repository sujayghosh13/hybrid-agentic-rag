import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    import docx
except ImportError:
    docx = None

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.models import DocType, DocumentElement, RawDocument

logger = logging.getLogger(__name__)


class DocxLoader(BaseLoader):
    """Loader for Microsoft Word (.docx) documents preserving heading structure."""

    HEADING_STYLE_REGEX = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)

    def can_load(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in (".docx", ".doc")

    def load(self, filepath: Path) -> RawDocument:
        if docx is None:
            raise ImportError(
                "python-docx is not installed. Please install it with 'pip install python-docx' to load Word documents."
            )
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            doc = docx.Document(str(filepath))
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX file '{filepath.name}': {e}") from e

        elements = self.parse_docx(doc)
        return RawDocument(
            filepath=filepath,
            filename=filepath.name,
            doc_type=DocType.DOCX,
            elements=elements,
        )

    def parse_docx(self, doc: Any) -> List[DocumentElement]:
        """Extract paragraphs and tables with hierarchical heading metadata."""
        elements: List[DocumentElement] = []
        heading_stack: List[Tuple[int, str]] = []
        current_text_lines: List[str] = []

        def flush_current_text():
            nonlocal current_text_lines
            if current_text_lines:
                text_block = "\n".join(current_text_lines).strip()
                if text_block:
                    section_path = [title for _, title in heading_stack]
                    current_heading = heading_stack[-1][1] if heading_stack else None
                    elements.append(
                        DocumentElement(
                            text=text_block,
                            heading=current_heading,
                            section_path=section_path,
                            page_number=None,
                        )
                    )
                current_text_lines = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            style_name = (paragraph.style.name if paragraph.style else "").strip()
            match = self.HEADING_STYLE_REGEX.match(style_name)

            if match:
                flush_current_text()
                level = int(match.group(1))
                title = text

                # Maintain heading stack hierarchy
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, title))
            elif style_name.lower() == "title":
                flush_current_text()
                heading_stack = [(1, text)]
            else:
                current_text_lines.append(text)

        flush_current_text()

        # Extract table rows as structured text blocks
        for table in doc.tables:
            table_lines: List[str] = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    # Deduplicate adjacent cells if merged
                    deduped_cells: List[str] = []
                    for c in row_cells:
                        if not deduped_cells or c != deduped_cells[-1]:
                            deduped_cells.append(c)
                    table_lines.append(" | ".join(deduped_cells))

            if table_lines:
                section_path = [title for _, title in heading_stack]
                current_heading = heading_stack[-1][1] if heading_stack else None
                elements.append(
                    DocumentElement(
                        text="\n".join(table_lines),
                        heading=current_heading,
                        section_path=section_path,
                        page_number=None,
                    )
                )

        return elements
