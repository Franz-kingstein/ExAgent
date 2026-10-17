"""Trace schema tests: missing-info honesty, integrity rules, serialization."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trace_xai.core.trace import (
    EventType,
    ExecutionTrace,
    Provenance,
    ProvenanceSource,
    TraceEvent,
)


def _event(event_id="E1", sequence=1, **kw) -> TraceEvent:
    defaults = dict(
        event_id=event_id,
        sequence=sequence,
        task_id="TASK-1",
        agent_id="agent",
        event_type=EventType.TASK_RECEIVED,
        provenance=Provenance(source=ProvenanceSource.TASK),
    )
    defaults.update(kw)
    return TraceEvent(**defaults)


def _trace(events) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id="TRC-1", task_id="TASK-1", task_text="t", agent_id="agent", events=events
    )


def test_missing_fields_are_null_not_fabricated():
    ev = _event()
    for field in (
        "timestamp", "action", "tool_name", "tool_input", "tool_output",
        "state_before", "state_after", "requirement_reference",
        "expected_outcome", "actual_outcome", "success", "error",
        "related_event_id", "conflict_with_event_id",
    ):
        assert getattr(ev, field) is None, f"{field} should be explicitly null"
    assert ev.provenance.synthetic is True


def test_sequence_must_be_strictly_increasing():
    with pytest.raises(ValidationError, match="strictly increasing"):
        _trace([_event("E1", 1), _event("E2", 1)])


def test_event_ids_must_be_unique():
    with pytest.raises(ValidationError, match="unique"):
        _trace([_event("E1", 1), _event("E1", 2)])


def test_events_must_reference_trace_task_id():
    with pytest.raises(ValidationError, match="task_id"):
        _trace([_event("E1", 1, task_id="TASK-OTHER")])


def test_conflicting_results_are_representable_not_overwritten():
    first = _event("E1", 1, event_type=EventType.TOOL_RESULT, tool_output={"success": True})
    conflict = _event(
        "E2", 2, event_type=EventType.TOOL_RESULT, tool_output={"success": False},
        conflict_with_event_id="E1",
    )
    trace = _trace([first, conflict])
    assert len(trace.events) == 2  # first result preserved
    assert trace.events[1].conflict_with_event_id == "E1"


def test_final_outcome_reads_last_outcome_event():
    trace = _trace([
        _event("E1", 1),
        _event("E2", 2, event_type=EventType.OUTCOME_RECORDED,
               actual_outcome="OK", success=True),
        _event("E3", 3, event_type=EventType.OUTCOME_RECORDED,
               actual_outcome="LIMIT_EXCEEDED", success=False),
    ])
    assert trace.final_outcome == (False, "LIMIT_EXCEEDED")


def test_final_outcome_none_without_outcome_event():
    assert _trace([_event()]).final_outcome == (None, None)


def test_serialization_round_trip():
    trace = _trace([
        _event("E1", 1, timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc)),
        _event("E2", 2, event_type=EventType.OUTCOME_RECORDED,
               actual_outcome="OK", success=True),
    ])
    assert ExecutionTrace.from_json(trace.to_json()) == trace


def test_empty_trace_is_valid_but_has_no_outcome():
    trace = _trace([])
    assert trace.final_outcome == (None, None)
