# hybrid-agentic-rag

A hybrid agentic RAG system incorporating structure-aware document ingestion, dense vector search, sparse BM25 search, and Reciprocal Rank Fusion (RRF).

## Setup

### Prerequisites
- Python 3.11
- Docker & Docker Compose (for running local Qdrant vector database)

### Basic Installation Instructions

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd hybrid-agentic-rag
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   # source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment configuration:
   ```bash
   cp .env.example .env
   ```

---

## Phase 1: Document Ingestion

1. Place raw documentation (`.md`, `.html`, `.pdf`) into `data/raw/`.
2. Run document ingestion and structure-aware chunking pipeline:
   ```bash
   python scripts/ingest.py data/raw
   ```
   Outputs structured chunks to `data/processed/chunks.jsonl`.

---

## Phase 2: Hybrid Retrieval

### 1. Start Qdrant Vector Database
Start local Qdrant container using Docker Compose:
```bash
docker compose up -d
```

### 2. Build Dense & Sparse Search Indices
Build Qdrant dense vector index (`BAAI/bge-small-en-v1.5`) and BM25 sparse index from processed chunks:
```bash
python scripts/build_index.py
```

### 3. Run Test Suite
Run unit tests for ingestion and hybrid retrieval:
```bash
python -m pytest
```
