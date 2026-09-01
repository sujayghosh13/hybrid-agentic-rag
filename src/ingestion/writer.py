import json
from pathlib import Path
from typing import Iterable

from src.ingestion.models import Chunk


class JSONLWriter:
    """Writer service to dump Chunk objects to JSONL format."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def write_chunks(self, chunks: Iterable[Chunk], append: bool = False) -> int:
        """Write chunks to JSONL file. Returns total count of chunks written."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        count = 0
        with self.output_path.open(mode, encoding="utf-8") as f:
            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
                count += 1

        return count
