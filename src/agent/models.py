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
class AgentResponse:
    """Final output returned by LocalQwenAgent."""
    query: str
    answer: str
    retrieval_needed: bool = True
    sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    thought_process: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieval_needed": self.retrieval_needed,
            "sources": self.sources,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "thought_process": self.thought_process,
            "metadata": self.metadata,
        }
