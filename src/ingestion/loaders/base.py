from abc import ABC, abstractmethod
from pathlib import Path

from src.ingestion.models import RawDocument


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def can_load(self, filepath: Path) -> bool:
        """Return True if this loader can handle the specified file."""
        pass

    @abstractmethod
    def load(self, filepath: Path) -> RawDocument:
        """Parse raw file and return a RawDocument object containing DocumentElements."""
        pass
