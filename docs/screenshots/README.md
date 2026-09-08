# Application Interface Screenshots & Artifacts

This directory contains visual records of the running **Hybrid-Agentic-RAG** Streamlit interface, satisfying Capstone Requirement Section 23 ("Screenshots of the application").

---

## Committed Screenshot Artifacts

| File | Description | Capstone Demonstration Focus |
| :--- | :--- | :--- |
| [`01_main_interface.png`](file:///d:/hybrid-agentic-rag/docs/screenshots/01_main_interface.png) | **Main Interface & Status Overview** | Full application landing view showing live Ollama & API status, model specifications (Qwen3, BGE-small, Cross-Encoder), storage readiness, example query chips, and multi-format document upload zone. |
| [`02_rag_answer_sources.png`](file:///d:/hybrid-agentic-rag/docs/screenshots/02_rag_answer_sources.png) | **RAG Answer & Source Attribution** | Real generated answer for *"How does Docker bridge networking work?"* along with the execution stat strip and verified evidence source cards displaying chunk IDs, relevance scores, and source text snippets. |
| [`03_execution_details.png`](file:///d:/hybrid-agentic-rag/docs/screenshots/03_execution_details.png) | **Orchestration Trace & CRAG Grading** | Detailed view of the orchestration pipeline: CRAG Evidence Grade (`GOOD`), Retrieval Hops (`1 / 2`), CRAG Correction Status (`No`), total latency (`11.25s`), and the step-by-step query routing and rewriting trace. |

---

## Capturing / Regenerating Screenshots

To regenerate these screenshots automatically from the live application, run:
```powershell
python scripts/capture_screenshots.py
```

### Manual Capture Procedure:
1. Ensure the stack is running:
   ```powershell
   docker compose up -d
   # or local python app:
   streamlit run app.py
   ```
2. Navigate to `http://localhost:8501`.
3. Select an example query (e.g., *"How does Docker bridge networking work?"*) and click **Ask assistant**.
4. Capture screenshots using browser DevTools (`Ctrl+Shift+P` $\rightarrow$ *Capture screenshot*) and save to this directory.
