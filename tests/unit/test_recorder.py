"""Recorder tests: ordering, determinism, honest missing information."""

import pytest

from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.core.trace import ErrorInfo, EventType, ExecutionTrace


def _recorder(task_id="TASK-SYN-0001") -> ExecutionRecorder:
    return ExecutionRecorder(task_id=task_id, task_text="Transfer ₹1 to BOB-001", agent_id="a")


def test_recorder_assigns_monotonic_sequences_ids_and_timestamps():
    rec = _recorder()
    rec.record_task_received()
    inv = rec.record_tool_invocation("transfer", "transfer", {"amount_inr": 1})
    rec.record_tool_result("transfer", "transfer", {"success": True}, related_event_id=inv.event_id)
    rec.record_outcome("OK", success=True)

    trace = rec.finish()
    assert [e.sequence for e in trace.events] == [1, 2, 3, 4]
    stamps = [e.timestamp for e in trace.events]
    assert stamps == sorted(stamps) and all(s is not None for s in stamps)
    ids = [e.event_id for e in trace.events]
    assert len(set(ids)) == 4
    assert ids[0] == "TRC-TASK-SYN-0001-E001"


def test_result_links_to_its_invocation():
    rec = _recorder()
    rec.record_task_received()
    inv = rec.record_tool_invocation("transfer", "transfer", {})
    out = rec.record_tool_result("transfer", "transfer", {"success": True},
                                 related_event_id=inv.event_id)
    trace = rec.finish()
    assert trace.events[inv.sequence - 1].event_id == out.related_event_id


def test_requirement_event_records_reference_and_expectation():
    rec = _recorder()
    rec.record_task_received()
    ev = rec.record_requirement("REQ-LIMIT-001", expected_outcome="transfer <= 50000")
    trace = rec.finish()
    assert trace.events[ev.sequence - 1].requirement_reference == "REQ-LIMIT-001"
    assert trace.events[ev.sequence - 1].expected_outcome == "transfer <= 50000"


def test_unobserved_state_stays_null():
    rec = _recorder()
    rec.record_task_received()
    inv = rec.record_tool_invocation("transfer", "transfer", {"amount_inr": 1})
    rec.record_tool_result("transfer", "transfer", None, related_event_id=inv.event_id)
    event = rec.finish().events[2]
    assert event.state_before is None and event.state_after is None
    assert event.tool_output is None and event.success is None


def test_conflicting_result_is_appended_not_overwritten():
    rec = _recorder()
    rec.record_task_received()
    inv = rec.record_tool_invocation("transfer", "transfer", {})
    first = rec.record_tool_result("transfer", "transfer", {"success": True},
                                   related_event_id=inv.event_id)
    rec.record_tool_result(
        "transfer", "transfer", {"success": False, "reason_code": "LIMIT_EXCEEDED"},
        related_event_id=inv.event_id, conflict_with_event_id=first.event_id,
    )
    trace = rec.finish()
    results = [e for e in trace.events if e.event_type is EventType.TOOL_RESULT]
    assert len(results) == 2
    assert results[1].conflict_with_event_id == first.event_id


def test_error_info_is_structured():
    rec = _recorder()
    rec.record_task_received()
    rec.record_tool_result("transfer", "transfer", None,
                           error=ErrorInfo(error_type="Timeout", message="env timed out"))
    assert rec.finish().events[1].error.error_type == "Timeout"


def test_recorder_is_deterministic():
    def build() -> ExecutionTrace:
        rec = _recorder()
        rec.record_task_received()
        inv = rec.record_tool_invocation("transfer", "transfer", {"amount_inr": 1})
        rec.record_tool_result("transfer", "transfer", {"success": False},
                               related_event_id=inv.event_id)
        rec.record_outcome("INSUFFICIENT_FUNDS", success=False)
        return rec.finish()

    assert build().to_json() == build().to_json()


def test_finish_without_events_raises():
    with pytest.raises(RuntimeError, match="no events"):
        _recorder().finish()


def test_events_property_returns_copy():
    rec = _recorder()
    rec.record_task_received()
    rec.events.clear()
    assert len(rec.events) == 1
