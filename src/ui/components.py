import html
import math
import re
from typing import Any, Dict, List, Optional
import streamlit as st

from src.ui.api_client import (
    APIConnectionError,
    APIServerError,
    APITimeoutError,
    APIValidationError,
    BackendUnavailableError,
)


def format_markdown_text(text: str) -> str:
    """Format and clean markdown text to ensure code blocks, tables, and commands render cleanly in Streamlit."""
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Separate concatenated prompt commands (e.g. '$docker ...$docker ...' or '$ cmd1 $ cmd2')
    text = re.sub(r'(\$\s*[a-zA-Z0-9_\-]+[^\n$]+?)(\$\s*[a-zA-Z0-9_\-]+)', r'\1\n\2', text)

    lines = text.split("\n")
    formatted_lines: List[str] = []
    in_code_block = False
    indent_len = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for code fence
        if stripped.startswith("```"):
            if not in_code_block:
                # Opening code fence
                indent_len = len(line) - len(line.lstrip())
                if formatted_lines and formatted_lines[-1].strip() != "":
                    formatted_lines.append("")
                # Place fence at base margin so Streamlit parser treats it as top-level code block
                lang = stripped[3:].strip()
                formatted_lines.append(f"```{lang}")
                in_code_block = True
            else:
                # Closing code fence
                formatted_lines.append("```")
                in_code_block = False
                indent_len = 0
                if i + 1 < len(lines) and lines[i + 1].strip() != "":
                    formatted_lines.append("")
            continue

        if in_code_block:
            # If code was indented inside a list, strip the list indentation
            if indent_len > 0 and line.startswith(" " * indent_len):
                formatted_lines.append(line[indent_len:])
            else:
                formatted_lines.append(line)
        else:
            # Ensure blank line before markdown tables
            if (
                stripped.startswith("|")
                and formatted_lines
                and not formatted_lines[-1].strip().startswith("|")
                and formatted_lines[-1].strip() != ""
            ):
                formatted_lines.append("")

            # Ensure blank line before headings (#, ##, ###)
            if re.match(r"^#{1,6}\s", stripped) and formatted_lines and formatted_lines[-1].strip() != "":
                formatted_lines.append("")

            # Ensure blank line before horizontal rules (---)
            if stripped == "---" and formatted_lines and formatted_lines[-1].strip() != "":
                formatted_lines.append("")

            formatted_lines.append(line)

            # Ensure blank line after horizontal rules (---)
            if stripped == "---" and i + 1 < len(lines) and lines[i + 1].strip() != "":
                formatted_lines.append("")

    return "\n".join(formatted_lines)


def format_source_text(text: str) -> str:
    """Format raw retrieved source chunks so technical text, CLI outputs, and config blocks render cleanly."""
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Fix concatenated commands in raw text like '$docker ...$docker ...'
    text = re.sub(r'(\$\s*[a-zA-Z0-9_\-]+[^\n$]+?)(\$\s*[a-zA-Z0-9_\-]+)', r'\1\n\2', text)

    # Protect non-HTML angle brackets like <LOOPBACK,UP,LOWER_UP> or <container_id> from being eaten by HTML parser
    text = re.sub(r'<([A-Za-z0-9_\-,\s]+)>', r'&lt;\1&gt;', text)

    return format_markdown_text(text)


def render_header() -> None:
    """Render the compact product header with developer/infra badge row."""
    header_html = """
    <div class="har-header">
        <div class="har-header-top">
            <h1 class="har-title">HYBRID AGENTIC RAG</h1>
        </div>
        <p class="har-subtitle">Local retrieval and reasoning for technical documentation.</p>
        <div class="har-badge-row">
            <span class="har-badge"><span class="har-badge-dot"></span>Qwen3</span>
            <span class="har-badge"><span class="har-badge-dot"></span>Qdrant</span>
            <span class="har-badge"><span class="har-badge-dot"></span>BM25 + RRF</span>
            <span class="har-badge"><span class="har-badge-dot"></span>Cross-Encoder</span>
            <span class="har-badge"><span class="har-badge-dot"></span>CRAG</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_sidebar_status(
    health_data: Optional[Dict[str, Any]],
    connection_error: Optional[str] = None,
) -> None:
    """Render the consolidated system status and model specification panel in the sidebar."""
    with st.sidebar:
        if connection_error:
            status_panel_html = f"""
            <div class="har-sidebar-panel">
                <div class="har-sidebar-section-title">SYSTEM STATUS</div>
                <div class="har-status-table">
                    <div class="har-status-row">
                        <span class="har-status-name">
                            <span class="har-status-dot har-status-offline">●</span> API
                        </span>
                        <span class="har-status-tag har-status-offline">OFFLINE</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(status_panel_html, unsafe_allow_html=True)

            error_guidance = f"""
            <div class="har-error-banner">
                <div class="har-error-title">Backend Disconnected</div>
                <div class="har-error-msg">Target: <code>{html.escape(str(connection_error))}</code></div>
            </div>
            """
            st.markdown(error_guidance, unsafe_allow_html=True)
            st.caption("To start the backend server:")
            st.code(".venv\\Scripts\\python.exe -m uvicorn src.api.main:app --reload", language="powershell")
            return

        if health_data:
            readiness = health_data.get("readiness", {})
            bm25_ok = readiness.get("bm25_index_ready", False)
            qdrant_ok = readiness.get("qdrant_storage_ready", False)
            ollama_ok = readiness.get("ollama_reachable", False)
            models = health_data.get("models", {})
            api_version = health_data.get("version", "0.1.0")

            def get_status_dot_and_tag(is_ok: bool, warn: bool = False) -> tuple[str, str]:
                if is_ok:
                    return ("har-status-ready", "READY")
                if warn:
                    return ("har-status-warn", "UNREACHABLE")
                return ("har-status-offline", "MISSING")

            api_dot, api_tag = "har-status-ready", "READY"
            qdrant_dot, qdrant_tag = get_status_dot_and_tag(qdrant_ok)
            bm25_dot, bm25_tag = get_status_dot_and_tag(bm25_ok)
            ollama_dot, ollama_tag = get_status_dot_and_tag(ollama_ok, warn=True)

            status_panel_html = f"""
            <div class="har-sidebar-panel">
                <div class="har-sidebar-section-title">SYSTEM STATUS (v{html.escape(api_version)})</div>
                <div class="har-status-table">
                    <div class="har-status-row">
                        <span class="har-status-name">
                            <span class="har-status-dot {api_dot}">●</span> API
                        </span>
                        <span class="har-status-tag {api_dot}">{api_tag}</span>
                    </div>
                    <div class="har-status-row">
                        <span class="har-status-name">
                            <span class="har-status-dot {qdrant_dot}">●</span> QDRANT
                        </span>
                        <span class="har-status-tag {qdrant_dot}">{qdrant_tag}</span>
                    </div>
                    <div class="har-status-row">
                        <span class="har-status-name">
                            <span class="har-status-dot {bm25_dot}">●</span> BM25
                        </span>
                        <span class="har-status-tag {bm25_dot}">{bm25_tag}</span>
                    </div>
                    <div class="har-status-row">
                        <span class="har-status-name">
                            <span class="har-status-dot {ollama_dot}">●</span> OLLAMA
                        </span>
                        <span class="har-status-tag {ollama_dot}">{ollama_tag}</span>
                    </div>
                </div>

                <div class="har-sidebar-section-title" style="margin-top: 16px;">CONFIGURED MODELS</div>
                <div class="har-model-block">
                    <div class="har-model-label">LLM</div>
                    <div class="har-model-val">{html.escape(models.get('ollama_model', 'qwen3:1.7b'))}</div>
                </div>
                <div class="har-model-block">
                    <div class="har-model-label">Embedding</div>
                    <div class="har-model-val">{html.escape(models.get('embedding_model', 'BAAI/bge-small-en-v1.5'))}</div>
                </div>
                <div class="har-model-block">
                    <div class="har-model-label">Reranker</div>
                    <div class="har-model-val">{html.escape(models.get('reranker_model', 'ms-marco-MiniLM-L-6-v2'))}</div>
                </div>
            </div>
            """
            st.markdown(status_panel_html, unsafe_allow_html=True)

            if not ollama_ok:
                st.warning("Ollama is unreachable. Ensure the service is active: `ollama serve`")


def render_pipeline_loader() -> None:
    """Render a lightweight, honest pipeline orchestration progress indicator during query execution."""
    loader_html = """
    <div class="har-pipeline-loader">
        <div class="har-pipeline-loader-header">
            <div class="har-pipeline-spinner"></div>
            <span>PROCESSING QUERY &middot; AGENTIC RAG PIPELINE ACTIVE</span>
        </div>
        <div class="har-pipeline-stages">
            <span class="har-pipeline-stage">01 ROUTING</span>
            <span class="har-pipeline-arrow">&rarr;</span>
            <span class="har-pipeline-stage">02 HYBRID RETRIEVAL</span>
            <span class="har-pipeline-arrow">&rarr;</span>
            <span class="har-pipeline-stage">03 RERANKING</span>
            <span class="har-pipeline-arrow">&rarr;</span>
            <span class="har-pipeline-stage">04 EVIDENCE CHECK</span>
            <span class="har-pipeline-arrow">&rarr;</span>
            <span class="har-pipeline-stage">05 SYNTHESIS</span>
        </div>
    </div>
    """
    st.markdown(loader_html, unsafe_allow_html=True)


def render_answer(answer: str) -> None:
    """Render the synthesized grounded answer within a dedicated developer surface."""
    formatted_answer = format_markdown_text(answer)
    with st.container():
        badge_html = """
        <div style="margin-bottom: 8px;">
            <span class="har-answer-badge">ANSWER</span>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown(formatted_answer)


def render_orchestration_metadata(
    orchestration: Dict[str, Any],
    performance: Dict[str, Any],
    question: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """Render execution metrics in a stat strip and orchestration metadata as a technical trace log."""
    grade = (orchestration.get("final_evidence_grade") or "N/A").upper()
    hops = orchestration.get("hops_executed", 1)
    is_corrected = orchestration.get("is_corrected", False)
    latency_ms = performance.get("total_latency_ms", 0.0)

    # 1. Execution Metrics Stat Strip
    grade_class = "har-metric-neutral"
    if "GOOD" in grade:
        grade_class = "har-metric-good"
    elif "PARTIAL" in grade:
        grade_class = "har-metric-warn"
    elif "BAD" in grade or "NO" in grade:
        grade_class = "har-metric-bad"

    crag_val = "CORRECTED" if is_corrected else "NOT CORRECTED"
    crag_class = "har-metric-warn" if is_corrected else "har-metric-good"

    if latency_ms >= 1000.0:
        latency_str = f"{latency_ms / 1000.0:.2f}s"
    else:
        latency_str = f"{latency_ms:.1f}ms"

    stat_strip_html = f"""
    <div class="har-metrics-strip">
        <div class="har-metric-card">
            <div class="har-metric-label">EVIDENCE</div>
            <div class="har-metric-value {grade_class}">{html.escape(grade)}</div>
        </div>
        <div class="har-metric-card">
            <div class="har-metric-label">HOPS</div>
            <div class="har-metric-value">{hops} / 2</div>
        </div>
        <div class="har-metric-card">
            <div class="har-metric-label">CRAG</div>
            <div class="har-metric-value {crag_class}">{crag_val}</div>
        </div>
        <div class="har-metric-card">
            <div class="har-metric-label">LATENCY</div>
            <div class="har-metric-value">{html.escape(latency_str)}</div>
        </div>
    </div>
    """
    st.markdown(stat_strip_html, unsafe_allow_html=True)

    # 2. Orchestration Technical Trace Log
    with st.expander("ORCHESTRATION TRACE", expanded=False):
        retrieval_needed = orchestration.get("retrieval_needed", True)
        router_step = "technical_query (retrieval_needed=True)" if retrieval_needed else "direct_query (retrieval_needed=False)"
        original_query = question or "Submitted technical question"
        rewritten = orchestration.get("rewritten_queries", [])
        retrieval_detail = f"hybrid search (dense Qdrant + sparse BM25, RRF fusion, hops={hops})"
        if rewritten:
            hop_queries_str = " | ".join([f"Hop {i}: &quot;{html.escape(rq)}&quot;" for i, rq in enumerate(rewritten, 1)])
            retrieval_detail += f"<br><span style='color: var(--har-text-secondary);'>{hop_queries_str}</span>"

        evidence_detail = f"{html.escape(grade)} (is_corrected={is_corrected})"
        synthesis_model = model_name or "qwen3:1.7b"
        synthesis_detail = f"{html.escape(synthesis_model)} (grounded generation with attributed sources)"

        trace_html = f"""
        <div class="har-trace-log">
            <div class="har-trace-step">
                <span class="har-trace-num">01</span>
                <span class="har-trace-step-name">ROUTER</span>
                <span class="har-trace-content">{html.escape(router_step)}</span>
            </div>
            <div class="har-trace-connector">&darr;</div>
            <div class="har-trace-step">
                <span class="har-trace-num">02</span>
                <span class="har-trace-step-name">QUERY</span>
                <span class="har-trace-content">&quot;{html.escape(original_query)}&quot;</span>
            </div>
            <div class="har-trace-connector">&darr;</div>
            <div class="har-trace-step">
                <span class="har-trace-num">03</span>
                <span class="har-trace-step-name">RETRIEVAL</span>
                <span class="har-trace-content">{retrieval_detail}</span>
            </div>
            <div class="har-trace-connector">&darr;</div>
            <div class="har-trace-step">
                <span class="har-trace-num">04</span>
                <span class="har-trace-step-name">EVIDENCE</span>
                <span class="har-trace-content">{evidence_detail}</span>
            </div>
            <div class="har-trace-connector">&darr;</div>
            <div class="har-trace-step">
                <span class="har-trace-num">05</span>
                <span class="har-trace-step-name">SYNTHESIS</span>
                <span class="har-trace-content">{synthesis_detail}</span>
            </div>
        </div>
        """
        st.markdown(trace_html, unsafe_allow_html=True)


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render retrieved document chunks as technical evidence cards."""
    st.markdown(
        f"""
        <div style="margin-top: 24px; margin-bottom: 12px;">
            <span class="har-metric-label" style="font-size: 11px;">RETRIEVED SOURCES ({len(sources)})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not sources:
        st.info("No sources were retrieved or required for this response.")
        return

    for idx, s in enumerate(sources, 1):
        chunk_id = s.get("chunk_id", f"chunk_{idx}")
        source_doc = s.get("source", "Unknown document")
        score = s.get("rerank_score")
        text = s.get("text", "")

        score_badges_html = ""
        expander_score_label = ""
        if score is not None:
            rel_pct = (1.0 / (1.0 + math.exp(-score))) * 100.0
            score_badges_html = f"""
            <span class="har-source-badge har-source-score">RERANK {score:+.4f}</span>
            <span class="har-source-badge har-source-rel">RELEVANCE {rel_pct:.1f}%</span>
            """
            expander_score_label = f" | RERANK {score:+.4f} ({rel_pct:.1f}%)"

        expander_title = f"[{idx}] {chunk_id}{expander_score_label}"

        with st.expander(expander_title, expanded=(idx == 1)):
            card_header_html = f"""
            <div class="har-source-header">
                <span class="har-source-name">{html.escape(chunk_id)}</span>
                <div class="har-source-badges">
                    <span class="har-source-badge">SRC: {html.escape(source_doc)}</span>
                    {score_badges_html}
                </div>
            </div>
            """
            st.markdown(card_header_html, unsafe_allow_html=True)
            st.markdown(format_source_text(text))


def render_error(error: Exception) -> None:
    """Render informative, user-friendly error banners based on exception type."""
    if isinstance(error, APIConnectionError):
        st.error(
            """
            **FastAPI Backend Offline**
            
            Could not connect to the FastAPI backend service. To start the server:
            ```powershell
            .venv\\Scripts\\python.exe -m uvicorn src.api.main:app --reload
            ```
            """
        )
    elif isinstance(error, BackendUnavailableError):
        st.error(
            f"""
            **Local LLM Unavailable (HTTP 503)**
            
            {error.message}
            
            Please ensure Ollama is running locally:
            ```powershell
            ollama serve
            ```
            """
        )
    elif isinstance(error, APIValidationError):
        st.warning(f"**Validation Notice**: {error.message}")
    elif isinstance(error, APITimeoutError):
        st.error(
            f"""
            **Request Timed Out**
            
            {error.message}
            The local CPU inference took longer than the configured timeout threshold.
            """
        )
    elif isinstance(error, APIServerError):
        st.error(f"**Server Error**: {error.message}")
    else:
        st.error(f"**Unexpected Error**: {str(error)}")
