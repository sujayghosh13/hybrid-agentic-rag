import streamlit as st

from src.ui.api_client import APIConnectionError, RAGApiClient, RAGClientError
from src.ui.components import (
    render_answer,
    render_error,
    render_header,
    render_orchestration_metadata,
    render_sidebar_status,
    render_sources,
)


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
        page_title="Hybrid-Agentic RAG Assistant",
        page_icon="🤖",
        layout="wide",
    )

    client = RAGApiClient()
    init_session_state(client)

    # 1. Header
    render_header()

    # 2. Sidebar with manual refresh button
    with st.sidebar:
        if st.button("🔄 Refresh System Status", use_container_width=True):
            fetch_system_health(client)
            st.rerun()

    render_sidebar_status(
        st.session_state.health_data,
        st.session_state.health_error,
    )

    # 3. Question Form
    with st.form(key="query_form", clear_on_submit=False):
        question = st.text_area(
            "Ask a technical question about Docker or Kubernetes:",
            placeholder="e.g. How does Docker bridge networking work?",
            height=100,
        )

        col_submit, col_clear = st.columns([1, 5])
        with col_submit:
            submit_clicked = st.form_submit_button("🚀 Ask Assistant", use_container_width=True)

    if submit_clicked:
        clean_question = question.strip() if question else ""
        if not clean_question:
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Querying knowledge base & synthesizing answer with CRAG..."):
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

    # 4. Render Active Response if present
    if st.session_state.current_response:
        res = st.session_state.current_response
        render_answer(res.get("answer", ""))
        render_orchestration_metadata(
            res.get("orchestration", {}),
            res.get("performance", {}),
        )
        render_sources(res.get("sources", []))

    # 5. History and Clear option
    if st.session_state.history:
        st.divider()
        col_hist_header, col_hist_btn = st.columns([5, 1])
        with col_hist_header:
            st.subheader(f"💬 Session History ({len(st.session_state.history)})")
        with col_hist_btn:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.current_response = None
                st.rerun()

        with st.expander("View past questions in this session", expanded=False):
            for item in reversed(st.session_state.history):
                st.markdown(f"**Q:** *{item['question']}*")
                st.markdown(f"**A:** {item['answer']}")
                st.markdown("---")


if __name__ == "__main__":
    main()
