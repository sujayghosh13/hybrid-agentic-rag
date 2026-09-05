"""ASGI application alias for uvicorn src.api.app:app entrypoint."""

from src.api.main import app

__all__ = ["app"]
