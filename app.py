"""Root Streamlit Entrypoint for Hybrid Agentic RAG.

Delegates execution directly to the canonical Streamlit UI implementation in src/ui/app.py.
Enables running the application from the repository root via:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure repository root is at the head of sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.ui.app import main

if __name__ == "__main__":
    main()
