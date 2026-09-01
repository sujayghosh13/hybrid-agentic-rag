from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup, Tag

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.models import DocType, DocumentElement, RawDocument


class HTMLLoader(BaseLoader):
    """Loader for HTML (.html, .htm) files extracting headings and text blocks."""

    HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    def can_load(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in (".html", ".htm")

    def load(self, filepath: Path) -> RawDocument:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = filepath.read_text(encoding="utf-8", errors="replace")
        elements = self.parse_html(content)
        return RawDocument(
            filepath=filepath,
            filename=filepath.name,
            doc_type=DocType.HTML,
            elements=elements,
        )

    def parse_html(self, html_content: str) -> List[DocumentElement]:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "head"]):
            script_or_style.decompose()

        elements: List[DocumentElement] = []
        heading_stack: List[Tuple[int, str]] = []
        current_text_blocks: List[str] = []

        def flush_text():
            nonlocal current_text_blocks
            if current_text_blocks:
                joined = "\n".join(current_text_blocks).strip()
                if joined:
                    section_path = [title for _, title in heading_stack]
                    current_heading = heading_stack[-1][1] if heading_stack else None
                    elements.append(
                        DocumentElement(
                            text=joined,
                            heading=current_heading,
                            section_path=section_path,
                            page_number=None,
                        )
                    )
                current_text_blocks = []

        body = soup.body if soup.body else soup

        for child in body.descendants:
            if isinstance(child, Tag):
                tag_name = child.name.lower()
                if tag_name in self.HEADING_TAGS:
                    flush_text()
                    level = self.HEADING_TAGS[tag_name]
                    title = child.get_text(strip=True)
                    if title:
                        while heading_stack and heading_stack[-1][0] >= level:
                            heading_stack.pop()
                        heading_stack.append((level, title))
                elif tag_name in ("p", "div", "li", "pre", "code", "blockquote", "td", "th"):
                    # Direct text inside block elements
                    direct_text = child.get_text(strip=True)
                    if direct_text and not child.find_parents(list(self.HEADING_TAGS.keys())):
                        # Ensure we don't duplicate text from nested tags
                        if not any(isinstance(c, Tag) and c.name in ("p", "div", "li", "pre", "blockquote") for c in child.children):
                            current_text_blocks.append(direct_text)

        flush_text()

        # Fallback if no block tags matched
        if not elements:
            raw_text = soup.get_text(separator="\n", strip=True)
            if raw_text:
                elements.append(
                    DocumentElement(
                        text=raw_text,
                        heading=None,
                        section_path=[],
                        page_number=None,
                    )
                )

        return elements
