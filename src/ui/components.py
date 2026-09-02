from typing import Any, Dict, List, Optional
import streamlit as st

from src.ui.api_client import (
    APIConnectionError,
    APIServerError,
    APITimeoutError,
    APIValidationError,
    BackendUnavailableError,
)


def render_header() -> None:
    """Render the application header and description."""
    st.title("🤖 Hybrid-Agentic RAG Assistant")
    st.markdown(
        """
        **Offline-first Technical Documentation Assistant** powered by **Qwen3 (local)**,
        **Dense Vector (Qdrant) + Sparse BM25 (RRF)**, **Cross-Encoder Reranking**, and **Corrective RAG (CRAG)**.
        """
    )
    st.divider()


def render_sidebar_status(
    health_data: Optional[Dict[str, Any]],
    connection_error: Optional[str] = None,
) -> None:
    """Render the system health, component readiness, and startup guidance in the sidebar."""
    with st.sidebar:
        st.header("⚙️ System Status")

        if connection_error:
            st.error("🔴 **FastAPI Backend Offline**")
            st.warning(
                """
                **To start the backend server:**
                ```powershell
                .venv\\Scripts\\python.exe -m uvicorn src.api.main:app --reload
                ```
                """
            )
            st.caption(f"Target URL: `{connection_error}`")
            return

        if health_data:
            st.success("🟢 **FastAPI Connected**")
            st.caption(f"API Version: `{health_data.get('version', '0.1.0')}`")

            st.subheader("Component Readiness")
            readiness = health_data.get("readiness", {})

            bm25_ok = readiness.get("bm25_index_ready", False)
            qdrant_ok = readiness.get("qdrant_storage_ready", False)
            ollama_ok = readiness.get("ollama_reachable", False)

            st.markdown(f"- BM25 Index: {'✅ Ready' if bm25_ok else '❌ Missing'}")
            st.markdown(f"- Qdrant Storage: {'✅ Ready' if qdrant_ok else '❌ Missing'}")
            st.markdown(f"- Ollama Service: {'✅ Reachable' if ollama_ok else '⚠️ Unreachable'}")

            if not ollama_ok:
                st.warning(
                    """
                    **Ollama is unreachable!**
                    Verify Ollama is running:
                    ```powershell
                    ollama serve
                    ```
                    """
                )

            st.subheader("Configured Models")
            models = health_data.get("models", {})
            st.markdown(f"**LLM:** `{models.get('ollama_model', 'qwen3:4b')}`")
            st.markdown(f"**Embedding:** `{models.get('embedding_model', 'BGE-small')}`")
            st.markdown(f"**Reranker:** `{models.get('reranker_model', 'MS-MARCO MiniLM')}`")


def render_answer(answer: str) -> None:
    """Render the synthesized grounded answer."""
    st.subheader("💡 Answer")
    st.markdown(answer)


def render_orchestration_metadata(
    orchestration: Dict[str, Any],
    performance: Dict[str, Any],
) -> None:
    """Render execution metrics and CRAG orchestration metadata."""
    st.subheader("📊 Execution & Orchestration Details")

    col1, col2, col3, col4 = st.columns(4)

    grade = orchestration.get("final_evidence_grade") or "N/A"
    hops = orchestration.get("hops_executed", 1)
    is_corrected = orchestration.get("is_corrected", False)
    latency_ms = performance.get("total_latency_ms", 0.0)

    with col1:
        st.metric(label="Evidence Grade", value=grade)
    with col2:
        st.metric(label="Retrieval Hops", value=f"{hops} / 2")
    with col3:
        st.metric(label="CRAG Corrected", value="Yes" if is_corrected else "No")
    with col4:
        if latency_ms >= 1000.0:
            st.metric(label="Total Latency", value=f"{latency_ms / 1000.0:.2f} s")
        else:
            st.metric(label="Total Latency", value=f"{latency_ms:.1f} ms")

    rewritten = orchestration.get("rewritten_queries", [])
    if rewritten:
        with st.expander("🔍 Query Rewriting Traces"):
            for i, rq in enumerate(rewritten, 1):
                st.markdown(f"- **Hop {i} Query:** `{rq}`")


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render retrieved document chunks and attribution details."""
    st.subheader(f"📚 Retrieved Sources ({len(sources)})")

    if not sources:
        st.info("No sources were retrieved or required for this response.")
        return

    for idx, s in enumerate(sources, 1):
        chunk_id = s.get("chunk_id", f"chunk_{idx}")
        source_doc = s.get("source", "Unknown document")
        score = s.get("rerank_score")
        text = s.get("text", "")

        score_badge = f" — Score: `{score:.4f}`" if score is not None else ""
        header = f"[{idx}] {chunk_id} ({source_doc}){score_badge}"

        with st.expander(header, expanded=(idx == 1)):
            st.markdown(text)


def render_error(error: Exception) -> None:
    """Render informative, user-friendly error banners based on exception type."""
    if isinstance(error, APIConnectionError):
        st.error(
            """
            ⚠️ **Backend API Unreachable**
            
            Could not connect to the FastAPI backend. Please verify that the backend is running:
            ```powershell
            .venv\\Scripts\\python.exe -m uvicorn src.api.main:app --reload
            ```
            """
        )
    elif isinstance(error, BackendUnavailableError):
        st.error(
            f"""
            🔌 **Local LLM Unavailable (HTTP 503)**
            
            {error.message}
            
            Please ensure Ollama is running locally:
            ```powershell
            ollama serve
            ```
            """
        )
    elif isinstance(error, APIValidationError):
        st.warning(f"ℹ️ **Validation Notice**: {error.message}")
    elif isinstance(error, APITimeoutError):
        st.error(
            f"""
            ⏱️ **Request Timed Out**
            
            {error.message}
            The local CPU inference took longer than the configured timeout.
            """
        )
    elif isinstance(error, APIServerError):
        st.error(f"❌ **Server Error**: {error.message}")
    else:
        st.error(f"⚠️ **Unexpected Error**: {str(error)}")
