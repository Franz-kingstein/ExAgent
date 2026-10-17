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


# --------------------------------------------------------------------- Q2
class AnomalyType(str, Enum):
    """Interpretable trajectory anomaly types (rule-based detector)."""

    UNEXPECTED_TOOL_CALL = "UNEXPECTED_TOOL_CALL"
    UNEXPECTED_EVENT_ORDER = "UNEXPECTED_EVENT_ORDER"
    MISSING_EXPECTED_EVENT = "MISSING_EXPECTED_EVENT"
    CONFLICTING_TOOL_RESULTS = "CONFLICTING_TOOL_RESULTS"
    REPEATED_FAILED_ACTION = "REPEATED_FAILED_ACTION"
    EXCESSIVE_RETRY = "EXCESSIVE_RETRY"
    INCONSISTENT_STATE_TRANSITION = "INCONSISTENT_STATE_TRANSITION"
    ACTION_INCONSISTENT_WITH_STATE = "ACTION_INCONSISTENT_WITH_STATE"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Anomaly(BaseModel):
    """One detected trajectory anomaly, grounded in trace events."""

    anomaly_id: str = Field(pattern=r"^ANOM-\d{4}$")
    event_ids: list[str] = Field(min_length=1, description="Where in the trace it was detected")
    anomaly_type: AnomalyType
    evidence: list[EvidenceItem] = Field(default_factory=list)
    severity: Severity
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class DeviationCategory(str, Enum):
    """Top-level answer to 'what went wrong?'."""

    NONE = "NONE"                                  # nothing detected
    REQUIREMENT_VIOLATION = "REQUIREMENT_VIOLATION"  # (A)
    TRAJECTORY_ANOMALY = "TRAJECTORY_ANOMALY"        # (B)
    BOTH = "BOTH"                                    # (C)
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # (D)


class RelationLabel(str, Enum):
    """Association between anomalies and violations. Never a causal claim."""

    NONE = "none"
    CONFIRMED = "confirmed"   # same tool call in the trace
    POSSIBLE = "possible"     # same trace, different locations
    UNKNOWN = "unknown"       # cannot be determined from the evidence


class DeviationExplanation(BaseModel):
    """Q2 output: hybrid requirement-violation + trajectory-anomaly analysis."""

    task_id: str
    trace_id: str
    task_text: str
    agent_id: str
    action: str
    category: DeviationCategory
    requirement_violations: list[RequirementAssessment] = Field(default_factory=list)
    unknown_requirement_ids: list[str] = Field(
        default_factory=list, description="Requirements that could not be assessed"
    )
    anomalies: list[Anomaly] = Field(default_factory=list)
    relation: RelationLabel = RelationLabel.NONE
    combined_explanation: str
    method: str = "hybrid_requirement_and_trajectory_analysis_v1"
    llm_used: bool = False


# --------------------------------------------------------------------- Q3
class ActionSpec(BaseModel):
    """A (possibly hypothetical) banking action."""

    tool: str = "transfer"
    account_id: str | None = None
    beneficiary_id: str | None = None
    amount_inr: int | None = None


class PredictedOutcome(BaseModel):
    """Simulation result for an alternative action (never an executed claim)."""

    success: bool | None = Field(default=None, description="Null if evidence insufficient")
    reason_code: str | None = None
    requirement_statuses: dict[str, RequirementStatus] = Field(default_factory=dict)
    expected_state_change: str | None = None
    basis: str = "deterministic_simulation"


class ReplayRecord(BaseModel):
    """Explicit record of a *controlled replay* in an isolated sandbox.

    Present only when the alternative was actually executed; its execution is
    separate from the traced execution and is never blended into it.
    """

    performed: bool
    execution_success: bool | None = None
    reason_code: str | None = None
    state_change: dict | None = None
    transaction_id: str | None = None
    note: str


class RequirementComparisonItem(BaseModel):
    requirement_id: str
    actual_status: RequirementStatus
    alternative_status: RequirementStatus
    changed: bool


class CounterfactualImpact(str, Enum):
    IMPROVES = "improves"
    WORSENS = "worsens"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


class Counterfactual(BaseModel):
    cf_id: str = Field(pattern=r"^CF-\d{4}$")
    source: str = Field(description="'generated_minimal' (minimal edit) or 'provided'")
    original_action: ActionSpec
    alternative_action: ActionSpec
    changed_condition: str
    predicted_outcome: PredictedOutcome
    replay: ReplayRecord | None = Field(
        default=None, description="Null when replay was not performed/available"
    )
    requirement_comparison: list[RequirementComparisonItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    impact: CounterfactualImpact = CounterfactualImpact.UNKNOWN
    uncertainty: float = Field(ge=0.0, le=1.0)
    explanation: str


class CounterfactualExplanation(BaseModel):
    """Q3 output: what could have happened under minimal alternative actions."""

    task_id: str
    trace_id: str
    task_text: str
    agent_id: str
    action: str
    actual_summary: str
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list, description="Generation/evaluation limitations"
    )
    method: str = "controlled_counterfactual_replay_v1"
    llm_used: bool = False
