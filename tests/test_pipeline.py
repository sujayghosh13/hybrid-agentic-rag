import json
from pathlib import Path
import tempfile
import pytest

from src.ingestion.pipeline import IngestionPipeline


def test_ingestion_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = Path(tmp_dir) / "raw"
        input_dir.mkdir()

        # Create sample markdown file
        sample_md = input_dir / "sample.md"
        sample_md.write_text(
            "# Sample Document\n\nThis is a test markdown document for the ingestion pipeline.\n\n## Sub Section\n\nMore details here.",
            encoding="utf-8",
        )

        output_file = Path(tmp_dir) / "processed" / "chunks.jsonl"

        pipeline = IngestionPipeline()
        stats = pipeline.process_directory(input_dir=input_dir, output_file=output_file)

        assert stats["documents_processed"] == 1
        assert stats["total_chunks"] >= 1
        assert output_file.exists()

        # Read JSONL file lines
        lines = output_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == stats["total_chunks"]

        record = json.loads(lines[0])
        assert "id" in record
        assert "text" in record
        assert "source" in record
        assert "metadata" in record
        assert record["metadata"]["filename"] == "sample.md"
        assert record["metadata"]["doc_type"] == "markdown"
