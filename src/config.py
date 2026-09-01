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

    def __post_init__(self):
        if not self.qdrant_url:
            self.qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()
