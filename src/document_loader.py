"""Capstone Structural Compatibility Wrapper for Document Loaders.

Re-exports the core document ingestion loaders and factory from the modular
`src.ingestion.loaders` subpackage to satisfy capstone specification Section 22.
"""

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.docx_loader import DocxLoader
from src.ingestion.loaders.factory import LOADERS, get_loader
from src.ingestion.loaders.html_loader import HTMLLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.models import DocType, DocumentElement, RawDocument

__all__ = [
    "LOADERS",
    "BaseLoader",
    "DocType",
    "DocumentElement",
    "DocxLoader",
    "HTMLLoader",
    "MarkdownLoader",
    "PDFLoader",
    "RawDocument",
    "get_loader",
]
