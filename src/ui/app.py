import html
from pathlib import Path
import streamlit as st

from src.ui.api_client import APIConnectionError, RAGApiClient, RAGClientError
from src.ui.components import (
    format_markdown_text,
    render_answer,
    render_error,
    render_header,
    render_orchestration_metadata,
    render_pipeline_loader,
    render_sidebar_status,
    render_sources,
)


def load_custom_css() -> None:
    """Load centralized CSS styling file into the Streamlit document."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def init_session_state(client: RAGApiClient) -> None:
    """Initialize local Streamlit session state and execute initial health check once."""
    if "health_data" not in st.session_state:
        st.session_state.health_data = None
        st.session_state.health_error = None
        fetch_system_health(client)

    if "current_response" not in st.session_state:
        st.session_state.current_response = None

    if "history" not in st.session_state:
        st.session_state.history = []

    if "query_input" not in st.session_state:
        st.session_state.query_input = ""


def fetch_system_health(client: RAGApiClient) -> None:
    """Fetch system health from FastAPI and cache in session state."""
    try:
        health = client.check_health()
        st.session_state.health_data = health
        st.session_state.health_error = None
    except APIConnectionError:
        st.session_state.health_data = None
        st.session_state.health_error = client.base_url
    except Exception as e:
        st.session_state.health_data = None
        st.session_state.health_error = str(e)


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Hybrid Agentic RAG",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_custom_css()

    client = RAGApiClient()
    init_session_state(client)

    # 1. Product Header
    render_header()

    # 2. Sidebar with status panel and refresh action
    with st.sidebar:
        if st.button("REFRESH STATUS", use_container_width=True):
            fetch_system_health(client)
            st.rerun()

    render_sidebar_status(
        st.session_state.health_data,
        st.session_state.health_error,
    )

    # 3. Query Area
    st.markdown('<div class="har-query-header">ASK THE DOCUMENTATION</div>', unsafe_allow_html=True)

    with st.form(key="query_form", clear_on_submit=False):
        question = st.text_area(
            label="Ask the documentation",
            value=st.session_state.query_input,
            placeholder="e.g. How does Docker bridge networking work?",
            height=100,
            label_visibility="collapsed",
        )

        col_submit, col_pad = st.columns([1, 4])
        with col_submit:
            submit_clicked = st.form_submit_button("ASK ASSISTANT →", use_container_width=True)

    # Example question chips
    st.markdown('<div class="har-chips-label">EXAMPLE QUERIES</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("How does Docker bridge networking work?", key="chip_1", use_container_width=True):
            st.session_state.query_input = "How does Docker bridge networking work?"
            st.rerun()
    with c2:
        if st.button("What's the difference between Docker bridge networks?", key="chip_2", use_container_width=True):
            st.session_state.query_input = "What's the difference between Docker bridge networks?"
            st.rerun()
    with c3:
        if st.button("What happens when a container joins multiple networks?", key="chip_3", use_container_width=True):
            st.session_state.query_input = "What happens when a container joins multiple networks?"
            st.rerun()

    if submit_clicked:
        clean_question = question.strip() if question else ""
        if not clean_question:
            st.warning("Please enter a question before submitting.")
        else:
            # Sync session state input
            st.session_state.query_input = clean_question

            # Render lightweight orchestration progress indicator during blocking execution
            loader_slot = st.empty()
            with loader_slot.container():
                render_pipeline_loader()

            try:
                response_data = client.query_rag(clean_question)
                st.session_state.current_response = response_data
                st.session_state.history.append({
                    "question": clean_question,
                    "answer": response_data.get("answer", ""),
                })
            except RAGClientError as e:
                render_error(e)
            except Exception as e:
                st.error(f"Unexpected application error: {e}")
            finally:
                loader_slot.empty()

    # 4. Render Active Result
    if st.session_state.current_response:
        res = st.session_state.current_response

        # Extract LLM model name from health data if available
        model_name = None
        if st.session_state.health_data and "models" in st.session_state.health_data:
            model_name = st.session_state.health_data["models"].get("ollama_model")

        # Result container with surface background
        with st.container(border=True):
            render_answer(res.get("answer", ""))

        render_orchestration_metadata(
            orchestration=res.get("orchestration", {}),
            performance=res.get("performance", {}),
            question=res.get("question") or clean_question if "clean_question" in locals() else None,
            model_name=model_name,
        )

        render_sources(res.get("sources", []))

    # 5. Session History
    if st.session_state.history:
        st.markdown('<div style="margin-top: 32px; border-top: 1px solid var(--har-border); padding-top: 20px;"></div>', unsafe_allow_html=True)
        col_hist_header, col_hist_btn = st.columns([5, 1])
        with col_hist_header:
            st.markdown(
                f'<span class="har-metric-label" style="font-size: 11px;">SESSION HISTORY ({len(st.session_state.history)})</span>',
                unsafe_allow_html=True,
            )
        with col_hist_btn:
            if st.button("CLEAR HISTORY", use_container_width=True):
                st.session_state.history = []
                st.session_state.current_response = None
                st.session_state.query_input = ""
                st.rerun()

        with st.expander("PREVIOUS QUERIES & ANSWERS", expanded=False):
            for item in reversed(st.session_state.history):
                with st.container(border=True):
                    st.markdown(f"**Question:** `{item['question']}`")
                    st.markdown('<span class="har-answer-badge" style="margin-top: 8px; display: inline-block;">ANSWER</span>', unsafe_allow_html=True)
                    st.markdown(format_markdown_text(item["answer"]))


if __name__ == "__main__":
    main()
