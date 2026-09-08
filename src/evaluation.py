"""Capstone Structural Compatibility Wrapper for RAG Evaluation.

Re-exports evaluation runner and dataset loaders from `src.evaluation`
to satisfy capstone specification Section 22.
"""

from src.evaluation.dataset import load_benchmark_dataset, validate_benchmark_dataset
from src.evaluation.models import EvaluationReport, PerQueryEvalResult
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationReport",
    "EvaluationRunner",
    "PerQueryEvalResult",
    "load_benchmark_dataset",
    "validate_benchmark_dataset",
]
