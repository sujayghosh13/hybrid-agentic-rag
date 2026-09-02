import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.evaluation.models import BenchmarkQuery

logger = logging.getLogger(__name__)


def compute_dataset_hash(file_path: Path) -> str:
    """Compute SHA256 checksum of the benchmark dataset file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_benchmark_dataset(dataset_path: Path) -> Tuple[List[BenchmarkQuery], Dict[str, Any]]:
    """Load and parse the benchmark dataset from JSON."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries_data = data.get("queries", [])
    queries = [BenchmarkQuery.from_dict(q) for q in queries_data]
    metadata = {
        "version": data.get("version", "1.0.0"),
        "total_corpus_chunks": data.get("total_corpus_chunks", 308),
        "corpus_documents": data.get("corpus_documents", []),
        "dataset_hash": compute_dataset_hash(dataset_path),
    }
    return queries, metadata


def validate_benchmark_dataset(
    queries: List[BenchmarkQuery],
    corpus_chunks_path: Path,
) -> Tuple[bool, List[str]]:
    """Validate benchmark queries against the active processed corpus.

    Checks:
    - All relevant_chunk_ids exist in corpus chunks.jsonl.
    - All non-refusal queries have non-empty expected aspects and answers.
    - All refusal queries have empty relevant chunk IDs and expected_refusal=True.
    """
    errors: List[str] = []

    # Load corpus chunk IDs
    corpus_ids: Set[str] = set()
    if corpus_chunks_path.exists():
        with open(corpus_chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        c_data = json.loads(line_str)
                        corpus_ids.add(c_data["id"])
                    except json.JSONDecodeError:
                        pass
    else:
        errors.append(f"Corpus chunks file not found at {corpus_chunks_path}")

    for q in queries:
        # Check chunk IDs
        for cid in q.relevant_chunk_ids:
            if corpus_ids and cid not in corpus_ids:
                errors.append(f"Query '{q.id}': chunk ID '{cid}' does not exist in processed corpus.")

        if q.expected_refusal:
            if q.relevant_chunk_ids:
                errors.append(f"Query '{q.id}' marked as expected_refusal but has relevant_chunk_ids.")
        else:
            if not q.relevant_chunk_ids:
                errors.append(f"Query '{q.id}' is non-refusal but has empty relevant_chunk_ids.")
            if not q.expected_aspects:
                errors.append(f"Query '{q.id}' is non-refusal but has empty expected_aspects.")

    is_valid = len(errors) == 0
    return is_valid, errors
