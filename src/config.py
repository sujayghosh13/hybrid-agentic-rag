import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Ensure HuggingFace cache directory exists within project
hf_cache = Path("data/cache/huggingface").resolve()
hf_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(hf_cache))


@dataclass
class Settings:
    """Centralized application settings."""
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "hybrid_chunks")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    bm25_index_path: Path = Path(os.getenv("BM25_INDEX_PATH", "data/processed/bm25_index.pkl"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates_count: int = int(os.getenv("RERANK_CANDIDATES_COUNT", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    agent_temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    agent_max_hops: int = int(os.getenv("AGENT_MAX_HOPS", "2"))
    query_rewriter_enabled: bool = os.getenv("QUERY_REWRITER_ENABLED", "true").lower() in ("true", "1", "yes")
    sufficiency_check_enabled: bool = os.getenv("SUFFICIENCY_CHECK_ENABLED", "true").lower() in ("true", "1", "yes")
    crag_enabled: bool = os.getenv("CRAG_ENABLED", "true").lower() in ("true", "1", "yes")
    crag_min_rerank_score: float = float(os.getenv("CRAG_MIN_RERANK_SCORE", "-5.0"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    fastapi_base_url: str = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
    cors_origins: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
    )
    ui_query_timeout: float = float(os.getenv("UI_QUERY_TIMEOUT", "360.0"))

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated cors_origins string into a list of cleaned origin URLs."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def __post_init__(self):
        if not self.qdrant_url:
            self.qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()
