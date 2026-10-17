"""Structured execution-trace schema — the primary TRACE-XAI evidence source.

Design rules (docs/safety.md):

- **Observable execution only.** Events describe what was received, invoked,
  returned, and what state changed. There are no chain-of-thought or
  fabricated "reasoning" fields.
- **Missing information stays missing.** Optional fields are ``None`` (JSON
  null) when the information genuinely was not observed — e.g. no tool
  output, unknown expected outcome, no requirement reference, unavailable
  state. Nothing is defaulted or invented.
- **Conflicts are representable.** Multiple ``TOOL_RESULT`` events may link
  to one invocation; a contradicting result marks ``conflict_with_event_id``
  instead of silently overwriting the first.

Event-type mapping to the conceptual event flow (task transfer):
TASK_RECEIVED → ``TASK_RECEIVED``; REQUIREMENT_IDENTIFIED →
``REQUIREMENT_IDENTIFIED`` (null until the requirements engine exists);
BENEFICIARY_CHECK/BENEFICIARY_VERIFIED → ``TOOL_INVOCATION``/``TOOL_RESULT``
for ``check_beneficiary``; TRANSFER_REQUESTED/TRANSFER_EXECUTED →
``TOOL_INVOCATION``/``TOOL_RESULT`` for ``transfer``; OUTCOME_RECORDED →
``OUTCOME_RECORDED``. We only record steps that actually happened — the
simulated environment enforces beneficiary verification inside the transfer
tool, so no synthetic "verification event" is fabricated.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class EventType(str, Enum):
    """Structural event types. Domain meaning comes from ``action``/``tool_name``."""

    TASK_RECEIVED = "TASK_RECEIVED"
    REQUIREMENT_IDENTIFIED = "REQUIREMENT_IDENTIFIED"
    TOOL_INVOCATION = "TOOL_INVOCATION"
    TOOL_RESULT = "TOOL_RESULT"
    OUTCOME_RECORDED = "OUTCOME_RECORDED"


class ProvenanceSource(str, Enum):
    """Where an event's content originates from."""

    TASK = "TASK"          # the natural-language task text
    AGENT = "AGENT"        # the agent's recorded decision/outcome
    ENVIRONMENT = "ENVIRONMENT"  # simulated bank tool outputs / state
    REQUIREMENT = "REQUIREMENT"  # requirement registry (Phase 3+)
    SYSTEM = "SYSTEM"      # recorder/infrastructure


class Provenance(BaseModel):
    """Attribution of an event: who produced it, and whether it is synthetic."""

    source: ProvenanceSource
    source_id: str | None = Field(default=None, description="e.g. agent name, tool name")
    synthetic: bool = Field(
        default=True,
        description="Safety flag: all TRACE-XAI traces derive from synthetic data",
    )


class ErrorInfo(BaseModel):
    """Structured error information for exceptional (crashing) failures.

    Rejected-but-handled tool results (e.g. LIMIT_EXCEEDED) are *not* errors:
    they are ``success=False`` results with a reason code and null error.
    """

    error_type: str
    message: str
    detail: dict | None = None


class TraceEvent(BaseModel):
    """One observable execution event."""

    event_id: str = Field(description="Unique within the trace, e.g. TRC-SYN-0001-E002")
    sequence: int = Field(ge=1, description="Monotonic ordering within the trace")
    timestamp: datetime | None = Field(
        default=None, description="Logical-clock time (never wall time); null if untracked"
    )
    task_id: str
    agent_id: str
    event_type: EventType
    action: str | None = Field(default=None, description="Semantic action, e.g. 'transfer'")
    tool_name: str | None = None
    tool_input: dict | None = Field(default=None, description="Observed input; null if not captured")
    tool_output: dict | None = Field(default=None, description="Observed output; null if not captured")
    state_before: dict | None = Field(default=None, description="State summary before; null if unavailable")
    state_after: dict | None = Field(default=None, description="State summary after; null if unavailable")
    requirement_reference: str | None = Field(default=None, description="Requirement id if identified; else null")
    expected_outcome: str | None = Field(default=None, description="Expected outcome if known; else null")
    actual_outcome: str | None = Field(default=None, description="Observed outcome/reason code, if any")
    success: bool | None = Field(default=None, description="Null when genuinely unknown")
    error: ErrorInfo | None = None
    provenance: Provenance
    related_event_id: str | None = Field(
        default=None, description="TOOL_RESULT -> its TOOL_INVOCATION event"
    )
    conflict_with_event_id: str | None = Field(
        default=None, description="Marks this event as contradicting another (kept, not overwritten)"
    )


class ExecutionTrace(BaseModel):
    """A complete, ordered trace of one agent execution."""

    trace_id: str
    task_id: str
    task_text: str
    agent_id: str
    started_at: datetime | None = Field(
        default=None, description="Logical-clock start; null if not tracked"
    )
    events: list[TraceEvent] = Field(default_factory=list)
    synthetic: bool = True

    @model_validator(mode="after")
    def _check_integrity(self) -> "ExecutionTrace":
        seqs = [e.sequence for e in self.events]
        if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
            raise ValueError("event sequences must be strictly increasing")
        ids = [e.event_id for e in self.events]
        if len(set(ids)) != len(ids):
            raise ValueError("event ids must be unique within a trace")
        if any(e.task_id != self.task_id for e in self.events):
            raise ValueError("all events must reference the trace's task_id")
        return self

    # ----------------------------------------------------------- conveniences
    @property
    def final_outcome(self) -> tuple[bool | None, str | None]:
        """(success, actual_outcome) from the last OUTCOME_RECORDED event."""
        for event in reversed(self.events):
            if event.event_type is EventType.OUTCOME_RECORDED:
                return event.success, event.actual_outcome
        return None, None

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "ExecutionTrace":
        return cls.model_validate_json(data)
