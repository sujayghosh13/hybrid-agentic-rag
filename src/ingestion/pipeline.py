import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.loaders.factory import get_loader
from src.ingestion.models import Chunk
from src.ingestion.writer import JSONLWriter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document loading, structure-aware chunking, and JSONL export."""

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".pdf"}

    def __init__(
        self,
        chunker: Optional[StructureAwareChunker] = None,
    ):
        self.chunker = chunker or StructureAwareChunker()

    def process_file(self, filepath: Path) -> List[Chunk]:
        """Process a single file through loader and chunker."""
        filepath = Path(filepath)
        loader = get_loader(filepath)
        doc = loader.load(filepath)
        chunks = self.chunker.chunk_document(doc)
        return chunks

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_file: Union[str, Path],
    ) -> Dict[str, Union[int, str]]:
        """Process all supported documents in input_dir and save chunks to output_file."""
        input_path = Path(input_dir)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

        files_to_process: List[Path] = []
        if input_path.is_file():
            if input_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files_to_process.append(input_path)
        else:
            for p in sorted(input_path.rglob("*")):
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files_to_process.append(p)

        all_chunks: List[Chunk] = []
        processed_docs = 0

        for file in files_to_process:
            try:
                chunks = self.process_file(file)
                all_chunks.extend(chunks)
                processed_docs += 1
                logger.info(f"Processed '{file.name}': generated {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to process file '{file}': {e}", exc_info=True)

        writer = JSONLWriter(output_path)
        chunks_written = writer.write_chunks(all_chunks)

        return {
            "documents_processed": processed_docs,
            "total_chunks": chunks_written,
            "output_path": str(output_path),
        }
