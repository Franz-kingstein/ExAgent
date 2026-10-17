"""ExecutionRecorder: builds an ExecutionTrace with guaranteed ordering.

The recorder is the single writer of trace events. It assigns ``sequence``
numbers, deterministic event IDs, and logical-clock timestamps (never wall
time), so identical runs produce identical traces.

Callers pass in *observed* data only. Anything not observed must be left as
None — the recorder never fills gaps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    Provenance,
    ProvenanceSource,
    TraceEvent,
)

DEFAULT_CLOCK_START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

_DEFAULT_PROVENANCE = {
    EventType.TASK_RECEIVED: ProvenanceSource.TASK,
    EventType.REQUIREMENT_IDENTIFIED: ProvenanceSource.REQUIREMENT,
    EventType.TOOL_INVOCATION: ProvenanceSource.ENVIRONMENT,
    EventType.TOOL_RESULT: ProvenanceSource.ENVIRONMENT,
    EventType.OUTCOME_RECORDED: ProvenanceSource.AGENT,
}


class ExecutionRecorder:
    """Append-only recorder for one agent execution."""

    def __init__(
        self,
        task_id: str,
        task_text: str,
        agent_id: str,
        clock_start: datetime | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._task_id = task_id
        self._task_text = task_text
        self._agent_id = agent_id
        self._clock_start = clock_start or DEFAULT_CLOCK_START
        self._trace_id = trace_id or f"TRC-{task_id}"
        self._events: list[TraceEvent] = []

    # ------------------------------------------------------------- properties
    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    # --------------------------------------------------------------- records
    def record_task_received(self) -> TraceEvent:
        return self._emit(EventType.TASK_RECEIVED)

    def record_requirement(
        self, requirement_reference: str, expected_outcome: str | None = None
    ) -> TraceEvent:
        """Record an identified requirement (Phase 3+ fills real references)."""
        return self._emit(
            EventType.REQUIREMENT_IDENTIFIED,
            requirement_reference=requirement_reference,
            expected_outcome=expected_outcome,
            provenance=Provenance(source=ProvenanceSource.REQUIREMENT, source_id=requirement_reference),
        )

    def record_tool_invocation(
        self,
        action: str,
        tool_name: str,
        tool_input: dict | None = None,
        state_before: dict | None = None,
    ) -> TraceEvent:
        """Record an observed tool invocation. ``state_before``: null if unavailable."""
        return self._emit(
            EventType.TOOL_INVOCATION,
            action=action,
            tool_name=tool_name,
            tool_input=tool_input,
            state_before=state_before,
            provenance=Provenance(source=ProvenanceSource.ENVIRONMENT, source_id=tool_name),
        )

    def record_tool_result(
        self,
        action: str,
        tool_name: str,
        tool_output: dict | None,
        state_after: dict | None = None,
        success: bool | None = None,
        error: ErrorInfo | None = None,
        related_event_id: str | None = None,
        conflict_with_event_id: str | None = None,
    ) -> TraceEvent:
        """Record an observed tool result. ``tool_output``/``state_after``: null
        if genuinely not observed — never fabricated."""
        return self._emit(
            EventType.TOOL_RESULT,
            action=action,
            tool_name=tool_name,
            tool_output=tool_output,
            state_after=state_after,
            success=success,
            error=error,
            related_event_id=related_event_id,
            conflict_with_event_id=conflict_with_event_id,
            provenance=Provenance(source=ProvenanceSource.ENVIRONMENT, source_id=tool_name),
        )

    def record_outcome(
        self,
        actual_outcome: str,
        success: bool,
        expected_outcome: str | None = None,
    ) -> TraceEvent:
        """Record the final observed outcome. ``expected_outcome``: null if unknown."""
        return self._emit(
            EventType.OUTCOME_RECORDED,
            actual_outcome=actual_outcome,
            expected_outcome=expected_outcome,
            success=success,
        )

    def finish(self) -> ExecutionTrace:
        """Seal the recorder into an immutable ExecutionTrace."""
        if not self._events:
            raise RuntimeError("cannot finish a recorder with no events")
        return ExecutionTrace(
            trace_id=self._trace_id,
            task_id=self._task_id,
            task_text=self._task_text,
            agent_id=self._agent_id,
            started_at=self._clock_start,
            events=list(self._events),
        )

    # ------------------------------------------------------------- internals
    def _emit(self, event_type: EventType, **fields) -> TraceEvent:
        sequence = len(self._events) + 1
        provenance = fields.pop("provenance", None) or Provenance(
            source=_DEFAULT_PROVENANCE[event_type], source_id=self._agent_id
        )
        event = TraceEvent(
            event_id=f"{self._trace_id}-E{sequence:03d}",
            sequence=sequence,
            timestamp=self._clock_start + timedelta(seconds=sequence),
            task_id=self._task_id,
            agent_id=self._agent_id,
            event_type=event_type,
            provenance=provenance,
            **fields,
        )
        self._events.append(event)
        return event
