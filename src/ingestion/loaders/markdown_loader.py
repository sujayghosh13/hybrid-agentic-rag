import re
from pathlib import Path
from typing import List, Tuple

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.models import DocType, DocumentElement, RawDocument


class MarkdownLoader(BaseLoader):
    """Loader for Markdown (.md, .markdown) files preserving header structure."""

    HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$")

    def can_load(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in (".md", ".markdown")

    def load(self, filepath: Path) -> RawDocument:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = filepath.read_text(encoding="utf-8")
        elements = self.parse_markdown(content)
        return RawDocument(
            filepath=filepath,
            filename=filepath.name,
            doc_type=DocType.MARKDOWN,
            elements=elements,
        )

    def parse_markdown(self, content: str) -> List[DocumentElement]:
        """Parse markdown string into DocumentElement objects with heading hierarchy."""
        lines = content.splitlines()
        elements: List[DocumentElement] = []

        # Heading stack stores (level, title) tuples
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

        for line in lines:
            match = self.HEADING_REGEX.match(line.strip())
            if match:
                flush_current_text()
                level = len(match.group(1))
                title = match.group(2).strip()

                # Pop headings of equal or deeper level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, title))
            else:
                current_text_lines.append(line)

        flush_current_text()

        # Fallback if document is completely empty
        if not elements and content.strip():
            elements.append(
                DocumentElement(
                    text=content.strip(),
                    heading=None,
                    section_path=[],
                    page_number=None,
                )
            )

        return elements
