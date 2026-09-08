"""Capstone Structural Compatibility Wrapper for the RAG Pipeline.

Re-exports the core agentic RAG orchestrator from `src.agent`
to satisfy capstone specification Section 22.
"""

from src.agent.agent import LocalQwenAgent
from src.agent.models import AgentResponse, AgentStep, HopTrace, ToolCall
from src.agent.tools import HybridSearchTool

__all__ = [
    "AgentResponse",
    "AgentStep",
    "HopTrace",
    "HybridSearchTool",
    "LocalQwenAgent",
    "ToolCall",
]
