# Hybrid Agentic RAG - UI Design Notes & Architecture

This document details the visual identity, UX decisions, and styling architecture implemented for the **Hybrid Agentic RAG** developer/infrastructure interface.

---

## 1. Visual Identity & Design System

The visual identity was created to feel appropriate for a developer/infrastructure product (similar to Kubernetes dashboards, Grafana, Docker CLI tools, Datadog, or Honeycomb) rather than a generic SaaS chatbot or basic Streamlit prototype.

### Exact Color Palette

| Token Name | Hex Value | Usage / Semantic Role |
| :--- | :--- | :--- |
| **Background** | `#0B0F14` | Global application canvas background (`.stApp`) |
| **Surface** | `#111820` | Base content surfaces, cards, containers, sidebar |
| **Elevated Surface** | `#151E27` | Metric cards, badges, code chips, input fields |
| **Border Primary** | `#1E293B` | Card contours, dividers, module outlines |
| **Border Subtle** | `#16202C` | Internal row separators and subtle borders |
| **Primary Text** | `#E6EDF3` | Headings, questions, primary answer content |
| **Secondary Text** | `#8B98A7` | Metadata labels, descriptions, secondary values |
| **Muted Text** | `#576575` | Captions, hints, subtle section headers |
| **Accent (Cyan)** | `#4FB3BF` | Brand color, interactive focus states, primary buttons, badges |
| **Accent Hover** | `#63C5D1` | Button hover state, interactive highlights |
| **Success (Green)** | `#69B578` | System ready dots, "GOOD" evidence grade, high relevance |
| **Warning (Amber)** | `#D6A85F` | "PARTIAL" evidence grade, CRAG corrected flag, unreachable warnings |
| **Error (Red)** | `#D46A6A` | "BAD" evidence grade, backend offline indicator, critical alerts |

### Typography System

The application is **strictly offline-first**. It contains zero dependencies on external font CDNs or Google Fonts.

- **System Sans Stack**:
  `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`
  Used for normal paragraph copy, section headings, and conversational responses.
- **Monospace Stack**:
  `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace`
  Used for model identifiers, technical metrics, source file paths, chunk IDs, rerank scores, latency values, and orchestration execution logs.

### Spacing Scale

A strict 4px/8px-based spacing scale is used throughout the CSS:
- `4px` (`--har-space-1`): Micro gaps, inline badge padding
- `8px` (`--har-space-2`): Element margins, pill padding, row gaps
- `12px` (`--har-space-3`): Metric card padding, compact separators
- `16px` (`--har-space-4`): Panel padding, card interiors, gutter spacing
- `24px` (`--har-space-6`): Primary container padding, section vertical rhythm
- `32px` (`--har-space-8`): Major layout boundaries and dividers

### Border Radius Scale

- Small (`4px`): Badges, inline code chips, model tags
- Medium (`8px`): Stat cards, source evidence cards, trace container, textarea
- Large (`10px`): Main answer card, primary surface containers

---

## 2. Component Design Rationales

### Header
**Why it was redesigned:**
The previous header used a generic emoji (`🤖 Hybrid-Agentic RAG Assistant`) with standard centered Streamlit markdown. In developer tools, emojis as the primary visual anchor cheapen the technical credibility.
The redesigned header:
- Presents a crisp, tracked title: `HYBRID AGENTIC RAG`
- Features a clear value proposition: *"Local retrieval and reasoning for technical documentation."*
- Introduces an intentional technology badge row (`Qwen3`, `Qdrant`, `BM25 + RRF`, `Cross-Encoder`, `CRAG`) that immediately informs engineers about the underlying architecture and offline retrieval stack.

### Sidebar (System Status & Models Consolidation)
**Why it was consolidated:**
In the original design, system health and model configurations were scattered into separate sections with standard Streamlit emoji headers.
The redesigned sidebar:
- Combines `SYSTEM STATUS` and `CONFIGURED MODELS` into a single, cohesive observability panel (`.har-sidebar-panel`).
- Replaces emojis with technical status indicators (`● READY`, `● OFFLINE`, `● UNREACHABLE`, `● MISSING`) with strict color-coding (`#69B578`, `#D46A6A`, `#D6A85F`).
- Formats model names (`qwen3:1.7b`, `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) in monospace blocks.
- Retains all startup troubleshooting instructions, API versioning, and refresh actions.

### Query Area & Example Chips
**Why it is visually emphasized:**
The query interface is the primary interaction point.
- Styled with an elevated surface (`#151E27`), crisp border (`#1E293B`), and an accent focus ring (`#4FB3BF`).
- Primary action button `ASK ASSISTANT →` has strong visual weight, high contrast, and responsive hover transitions.
- Interactive question chips (`"How does Docker bridge networking work?"`, `"What's the difference between Docker bridge networks?"`, `"What happens when a container joins multiple networks?"`) allow users to immediately run representative technical queries with zero typing, interacting directly through `st.session_state` without altering any backend logic.

### Orchestration Pipeline Loading State
**Why it is structured as a pipeline indicator:**
During local LLM inference, responses take several seconds depending on hardware. A generic spinning wheel gives no feedback on what is occurring.
The loader displays the true pipeline stages:
`01 ROUTING → 02 HYBRID RETRIEVAL → 03 RERANKING → 04 EVIDENCE CHECK → 05 SYNTHESIS`
It remains honest: it does not claim individual stages have finished when the API is a single blocking call, but clearly presents the active agentic workflow.

### Answer Card
**Why it has a dedicated result surface:**
The synthesized answer is the centerpiece of the user journey.
- Wrapped in a dedicated result surface (`#111820`) with a distinct `ANSWER` badge.
- Typography is tuned for readability (`15px`, `1.65` line height).
- Inline entity and command references (e.g. `docker0`, `--link`, `bridge`) render as stylized technical chips (`#151E27` background, `#4FB3BF` text, `#1E293B` border) rather than raw unstyled markdown.
- Code blocks, command lines, and tables are preserved with clean borders and contrast.

### Execution Metrics Dashboard Strip
**Why it is presented as a dashboard strip:**
Agentic RAG is differentiated by its dynamic retrieval and self-correction telemetry. Default Streamlit `st.metric` cards are bulky and lack semantic color-coding.
The 4-card metric strip provides an at-a-glance observability summary:
1. **EVIDENCE**: Dynamically color-coded (`GOOD` in green `#69B578`, `PARTIAL` in amber `#D6A85F`, `BAD` in red `#D46A6A`).
2. **HOPS**: Displays `hops_executed / 2` in monospace.
3. **CRAG**: Indicates whether the corrective loop triggered (`CORRECTED` or `NOT CORRECTED`).
4. **LATENCY**: Formatted in monospace seconds (`7.17s`) or milliseconds (`142.5ms`).

### Orchestration Trace Log
**Why it is presented as a technical log:**
Rather than a loose list of queries, the collapsible trace log is formatted like a structured terminal execution log:
- Numbered sequence (`01 ROUTER`, `02 QUERY`, `03 RETRIEVAL`, `04 EVIDENCE`, `05 SYNTHESIS`)
- Subtle directional connecting arrows (`↓`)
- Monospace styling displaying actual values (routing branch, user query, hybrid search parameters, evidence verification, and generator model).

### Sources Evidence Cards
**Why sources are presented as evidence cards:**
Retrieved chunks are evidence supporting the answer.
Each source chunk displays:
- Document path / origin
- Unique chunk identifier (e.g. `docker-bridge-network.html_chunk_8`)
- Monospace badges for **Reranker Score** (e.g. `+3.3099`) and calculated **Relevance Percentage** (e.g. `96.5%`) via sigmoid normalization.
- Expandable clean text preview with angle-bracket protection for Linux network flags and command outputs.

### Session History
**Why history was redesigned:**
Maintains session history entries in clean surface cards with question tags and formatted answer blocks, alongside an explicit `CLEAR HISTORY` action.

---

## 3. Offline & Production Architecture

1. **No External CDNs**: No Google Fonts, Font Awesome, or CDN styles.
2. **Native Streamlit Integration**: Combines `.streamlit/config.toml` dark theme tokens with `src/ui/styles.css` for instant loading without theme flickering.
3. **Zero Backend Changes**: The UI consumes the unmodified `POST /query` and `GET /health` endpoints of the Phase 7 FastAPI backend.
