"""
Streamlit Application Host for Hybrid Agentic RAG.

Serves as the application host while rendering the canonical Claude HTML design
(hybrid-rag-frontend.html / frontend/index.html) with zero-margin overrides to match
the visual source of truth precisely.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running via Streamlit
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import streamlit.components.v1 as components

from src.ui.api_client import APIConnectionError, RAGApiClient, RAGClientError


def main() -> None:
    """Main Streamlit application entrypoint hosting the canonical visual design."""
    st.set_page_config(
        page_title="Hybrid Agentic RAG",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Apply CSS overrides to hide Streamlit chrome and allow full-bleed iframe rendering
    st.markdown(
        """
        <style>
            #MainMenu { visibility: hidden; }
            header { visibility: hidden; height: 0 !important; }
            footer { visibility: hidden; }
            .stApp {
                background-color: #0E1410 !important;
            }
            .block-container {
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
                padding-left: 0rem !important;
                padding-right: 0rem !important;
                max-width: 100% !important;
            }
            iframe {
                width: 100% !important;
                border: none !important;
                min-height: 100vh !important;
                display: block;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Locate canonical HTML template
    html_path = project_root / "frontend" / "index.html"
    if not html_path.exists():
        fallback_path = Path("d:/hybrid-rag-frontend.html")
        if fallback_path.exists():
            html_path = fallback_path

    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        components.html(html_content, height=1300, scrolling=True)
    else:
        st.error(f"Canonical frontend HTML not found at {html_path}")


if __name__ == "__main__":
    main()
