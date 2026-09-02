from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents an invocation of an agent tool."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result if not isinstance(self.result, list) else f"[{len(self.result)} items]",
        }


@dataclass
class AgentStep:
    """Represents a single step in the agent reasoning loop."""
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[str] = None


@dataclass
class HopTrace:
    """Records the execution trace and decision for a retrieval hop."""
    hop_index: int
    query_used: str
    candidates_retrieved: int
    reranked_chunks_added: int
    is_sufficient: bool
    missing_aspect: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "query_used": self.query_used,
            "candidates_retrieved": self.candidates_retrieved,
            "reranked_chunks_added": self.reranked_chunks_added,
            "is_sufficient": self.is_sufficient,
            "missing_aspect": self.missing_aspect,
        }


@dataclass
class AgentResponse:
    """Final output returned by LocalQwenAgent."""
    query: str
    answer: str
    retrieval_needed: bool = True
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    thought_process: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hop_traces: List[HopTrace] = field(default_factory=list)
    hops_executed: int = 1
    rewritten_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieval_needed": self.retrieval_needed,
            "sources": self.sources,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "thought_process": self.thought_process,
            "metadata": self.metadata,
            "hop_traces": [ht.to_dict() for ht in self.hop_traces],
            "hops_executed": self.hops_executed,
            "rewritten_queries": self.rewritten_queries,
        }
