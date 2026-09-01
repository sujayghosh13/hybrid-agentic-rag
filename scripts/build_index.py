#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import settings
from src.ingestion.models import Chunk, ChunkMetadata
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever


def load_chunks_from_jsonl(jsonl_path: Path) -> list:
    """Load Chunk objects from a JSONL file."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Processed chunks file not found: {jsonl_path}")

    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            meta_dict = data.get("metadata", {})
            metadata = ChunkMetadata(
                filename=meta_dict.get("filename", ""),
                doc_type=meta_dict.get("doc_type", ""),
                page_number=meta_dict.get("page_number"),
                section=meta_dict.get("section"),
                heading=meta_dict.get("heading"),
                chunk_index=meta_dict.get("chunk_index", 0),
                total_chunks=meta_dict.get("total_chunks", 0),
                token_count=meta_dict.get("token_count", 0),
                char_count=meta_dict.get("char_count", 0),
            )
            chunk = Chunk(
                id=data["id"],
                text=data["text"],
                source=data["source"],
                metadata=metadata,
            )
            chunks.append(chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Build dense Qdrant vector index and sparse BM25 index from chunks.jsonl"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/processed/chunks.jsonl",
        help="Input JSONL filepath (default: data/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=settings.qdrant_url,
        help=f"Qdrant service URL (default: {settings.qdrant_url})",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help=f"Qdrant collection name (default: {settings.qdrant_collection})",
    )
    parser.add_argument(
        "--bm25-out",
        default=str(settings.bm25_index_path),
        help=f"BM25 index output path (default: {settings.bm25_index_path})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    input_path = Path(args.input)
    print(f"Loading chunks from '{input_path}'...")
    chunks = load_chunks_from_jsonl(input_path)
    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found to index!")
        return

    # 1. Build Dense Vector Index in Qdrant
    print(f"\n[1/2] Building Dense Vector Index in Qdrant (Collection: '{args.collection}')...")
    dense_retriever = DenseRetriever(
        collection_name=args.collection,
        embedding_model_name=settings.embedding_model_name,
    )
    dense_count = dense_retriever.index_chunks(chunks)

    # 2. Build Sparse BM25 Index
    print(f"\n[2/2] Building Sparse BM25 Index ('{args.bm25_out}')...")
    sparse_retriever = BM25Retriever(index_path=Path(args.bm25_out))
    sparse_count = sparse_retriever.index_chunks(chunks, save_path=Path(args.bm25_out))

    print("\n" + "=" * 50)
    print("INDEX BUILDING COMPLETE!")
    print(f"  Total Chunks Processed:  {len(chunks)}")
    print(f"  Qdrant Dense Vector Points: {dense_count}")
    print(f"  BM25 Sparse Items:       {sparse_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
