#!/usr/bin/env python
"""Evaluation CLI entry point for Hybrid-Agentic RAG.

Usage:
    # Run fast retrieval & reranker benchmark (0 LLM calls):
    python scripts/run_evaluation.py --mode fast

    # Run full agentic evaluation with LocalQwenAgent:
    python scripts/run_evaluation.py --mode full
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure UTF-8 stdout and stderr on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.reporting import EvaluationReporter
from src.evaluation.runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="Hybrid-Agentic RAG Evaluation Harness")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="Evaluation mode: 'fast' (pure retrieval/reranker, 0 LLM calls) or 'full' (complete agentic pipeline).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/benchmark_dataset.json",
        help="Path to benchmark dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluation/results",
        help="Directory to store evaluation JSON and CSV artifacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    runner = EvaluationRunner(dataset_path=dataset_path)
    try:
        report = runner.run(mode=args.mode)
    except Exception as e:
        import traceback
        print("ERROR IN EVALUATION RUNNER:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # 1. Print formatted summary to console
    EvaluationReporter.print_summary(report)

    # 2. Export JSON and CSV artifacts
    json_path = EvaluationReporter.save_json(report, output_dir)
    csv_path = EvaluationReporter.save_csv(report, output_dir)

    print(f"Artifacts saved successfully:")
    print(f" - JSON Report : {json_path}")
    print(f" - CSV Results : {csv_path}")


if __name__ == "__main__":
    main()
