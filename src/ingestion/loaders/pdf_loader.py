from pathlib import Path
from typing import List

from pypdf import PdfReader

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.models import DocType, DocumentElement, RawDocument


class PDFLoader(BaseLoader):
    """Loader for PDF (.pdf) files preserving page numbers and page-level text."""

    def can_load(self, filepath: Path) -> bool:
        return filepath.suffix.lower() == ".pdf"

    def load(self, filepath: Path) -> RawDocument:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        elements = self.parse_pdf(filepath)
        return RawDocument(
            filepath=filepath,
            filename=filepath.name,
            doc_type=DocType.PDF,
            elements=elements,
        )

    def parse_pdf(self, filepath: Path) -> List[DocumentElement]:
        reader = PdfReader(str(filepath))
        elements: List[DocumentElement] = []

        for i, page in enumerate(reader.pages):
            page_number = i + 1  # 1-indexed page number
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            # Infer top line as heading candidate if short and title-case/uppercase
            first_line = lines[0] if lines else None
            heading = None
            if first_line and len(first_line) < 80 and (first_line.isupper() or first_line.istitle()):
                heading = first_line

            elements.append(
                DocumentElement(
                    text=text,
                    heading=heading,
                    section_path=[heading] if heading else [],
                    page_number=page_number,
                )
            )

        return elements
