#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is in Python path when executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingest and chunk raw technical documentation into structured JSONL."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/raw",
        help="Input folder containing PDF, Markdown, or HTML files (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/chunks.jsonl",
        help="Output JSONL filepath (default: data/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=300,
        help="Minimum target tokens per chunk (default: 300)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum target tokens per chunk (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="Chunk overlap fraction between 0.0 and 1.0 (default: 0.15)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    chunker = StructureAwareChunker(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap,
    )

    pipeline = IngestionPipeline(chunker=chunker)

    print(f"Starting ingestion from '{args.input_dir}'...")
    results = pipeline.process_directory(
        input_dir=args.input_dir,
        output_file=args.output,
    )

    print("\nIngestion Complete!")
    print(f"  Documents processed: {results['documents_processed']}")
    print(f"  Total chunks:        {results['total_chunks']}")
    print(f"  Output saved to:     {results['output_path']}")


if __name__ == "__main__":
    main()
