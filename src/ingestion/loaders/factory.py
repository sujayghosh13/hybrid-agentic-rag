from pathlib import Path
from typing import List

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.loaders.html_loader import HTMLLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader

LOADERS: List[BaseLoader] = [
    MarkdownLoader(),
    HTMLLoader(),
    PDFLoader(),
]


def get_loader(filepath: Path) -> BaseLoader:
    """Return the appropriate BaseLoader for the given filepath."""
    for loader in LOADERS:
        if loader.can_load(filepath):
            return loader
    raise ValueError(f"No supported loader found for file: {filepath} (extension: {filepath.suffix})")
