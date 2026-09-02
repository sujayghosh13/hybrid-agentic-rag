from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EvidenceGrade(str, Enum):
    """Classification grades for retrieved evidence quality."""
    GOOD = "GOOD"
    PARTIAL = "PARTIAL"
    BAD = "BAD"


@dataclass
class EvidenceEvaluation:
    """Represents the output of the CRAG evidence quality evaluator."""
    grade: EvidenceGrade
    missing_aspect: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grade": self.grade.value,
            "missing_aspect": self.missing_aspect,
            "reason": self.reason,
        }


@dataclass
class CorrectionTrace:
    """Trace record capturing a corrective RAG evaluation and action."""
    hop_index: int
    query_used: str
    evidence_grade: EvidenceGrade
    missing_aspect: Optional[str] = None
    reason: Optional[str] = None
    action_taken: str = ""
    candidates_retrieved: int = 0
    reranked_chunks_added: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "query_used": self.query_used,
            "evidence_grade": self.evidence_grade.value,
            "missing_aspect": self.missing_aspect,
            "reason": self.reason,
            "action_taken": self.action_taken,
            "candidates_retrieved": self.candidates_retrieved,
            "reranked_chunks_added": self.reranked_chunks_added,
        }
