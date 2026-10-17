"""Decision-explainability output models (Q1).

Every assessment is grounded in EvidenceItems that reference concrete events
and fields of an ExecutionTrace — nothing here can exist without a trace.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RequirementStatus(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"  # missing or conflicting evidence — never assumed satisfied


class EvidenceItem(BaseModel):
    """A pointer to one observed value inside a trace event."""

    event_id: str
    field: str = Field(description="Dotted path inside the event, e.g. tool_input.amount_inr")
    value: Any = Field(description="The observed value at that path")


class RequirementAssessment(BaseModel):
    requirement_id: str
    requirement_text: str
    relevant_action: str
    status: RequirementStatus
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    conflicting_evidence: list[EvidenceItem] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class DecisionExplanation(BaseModel):
    """Q1 output: why the observed action satisfied/violated each requirement."""

    task_id: str
    trace_id: str
    task_text: str
    agent_id: str
    action: str
    requirements: list[RequirementAssessment] = Field(default_factory=list)
    rationale: str = Field(description="Concise evidence-grounded rationale")
    method: str = "requirement_to_action_matching_v1"
    llm_used: bool = False
