"""Trajectory anomaly detector tests (interpretable rules, no ML)."""

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.core.trace import ExecutionTrace
from trace_xai.explain import AnomalyType, TrajectoryAnomalyDetector

STATE_OK = {
    "balances": {"ACC-SYN-001": 250000},
    "per_txn_limits": {"ACC-SYN-001": 50000},
    "beneficiaries_verified": {"BOB-001": True},
}

DETECTOR = TrajectoryAnomalyDetector()


def _recorder(trace_id, task="Transfer ₹10,000 to beneficiary BOB-001.") -> ExecutionRecorder:
    return ExecutionRecorder(
        task_id=f"TASK-{trace_id}", task_text=task, agent_id="manual-agent", trace_id=trace_id
    )


def _types(anomalies) -> list[AnomalyType]:
    return [a.anomaly_type for a in anomalies]


def test_normal_agent_trace_has_no_anomalies():
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")[1]
    assert DETECTOR.detect(trace) == []


def test_unexpected_tool_call():
    rec = _recorder("TRC-UNEXP")
    rec.record_task_received()
    inv = rec.record_tool_invocation("WIRE", "wire_internationally", {"amount_inr": 1})
    rec.record_tool_result("WIRE", "wire_internationally", {"success": True},
                           related_event_id=inv.event_id)
    rec.record_outcome("OK", success=True)
    anomalies = DETECTOR.detect(rec.finish())
    assert _types(anomalies) == [AnomalyType.UNEXPECTED_TOOL_CALL]
    assert anomalies[0].severity.value == "HIGH"
    assert len(anomalies[0].event_ids) == 2  # invocation + result, one anomaly
    assert all(e.value == "wire_internationally" for e in anomalies[0].evidence)


def test_repeated_failed_action_threshold():
    def trace_with_failures(n):
        rec = _recorder("TRC-RETRY")
        rec.record_task_received()
        for _ in range(n):
            inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                             {"beneficiary_id": "BOB-001", "amount_inr": 10000})
            rec.record_tool_result("TRANSFER", "transfer",
                                   {"success": False, "reason_code": "INTERNAL_ERROR"},
                                   success=False, related_event_id=inv.event_id)
        rec.record_outcome("INTERNAL_ERROR", success=False)
        return rec.finish()

    assert _types(DETECTOR.detect(trace_with_failures(2))) == [
        AnomalyType.REPEATED_FAILED_ACTION]
    assert _types(DETECTOR.detect(trace_with_failures(3))) == [AnomalyType.EXCESSIVE_RETRY]


def test_inconsistent_state_on_success():
    rec = _recorder("TRC-STATES")
    rec.record_task_received()
    inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                     {"beneficiary_id": "BOB-001", "amount_inr": 10000},
                                     state_before=STATE_OK)
    rec.record_tool_result(
        "TRANSFER", "transfer",
        {"success": True, "reason_code": "OK", "account_id": "ACC-SYN-001"},
        state_after={**STATE_OK, "balances": {"ACC-SYN-001": 250000}},  # unchanged!
        success=True, related_event_id=inv.event_id,
    )
    rec.record_outcome("OK", success=True)
    anomalies = DETECTOR.detect(rec.finish())
    assert _types(anomalies) == [AnomalyType.INCONSISTENT_STATE_TRANSITION]
    assert "expected 240000" in anomalies[0].explanation


def test_inconsistent_state_on_rejection():
    rec = _recorder("TRC-STATEF")
    rec.record_task_received()
    inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                     {"beneficiary_id": "BOB-001", "amount_inr": 10000},
                                     state_before=STATE_OK)
    rec.record_tool_result(
        "TRANSFER", "transfer", {"success": False, "reason_code": "LIMIT_EXCEEDED"},
        state_after={**STATE_OK, "balances": {"ACC-SYN-001": 240000}},  # funds moved!
        success=False, related_event_id=inv.event_id,
    )
    rec.record_outcome("LIMIT_EXCEEDED", success=False)
    anomalies = DETECTOR.detect(rec.finish())
    assert _types(anomalies) == [AnomalyType.INCONSISTENT_STATE_TRANSITION]
    assert "funds moved despite failure" in anomalies[0].explanation


def test_missing_outcome_event():
    rec = _recorder("TRC-NOOUT")
    rec.record_task_received()
    inv = rec.record_tool_invocation("BALANCE", "get_balance", {})
    rec.record_tool_result("BALANCE", "get_balance", {"success": True},
                           related_event_id=inv.event_id)
    anomalies = DETECTOR.detect(rec.finish())
    assert _types(anomalies) == [AnomalyType.MISSING_EXPECTED_EVENT]


def test_dangling_result_reference():
    rec = _recorder("TRC-DANGLE")
    rec.record_task_received()
    rec.record_tool_result("TRANSFER", "transfer", {"success": True, "reason_code": "OK"},
                           success=True, related_event_id="TRC-DANGLE-E999")
    rec.record_outcome("OK", success=True)
    anomalies = DETECTOR.detect(rec.finish())
    assert _types(anomalies) == [AnomalyType.UNEXPECTED_EVENT_ORDER]
    assert anomalies[0].evidence[0].value == "TRC-DANGLE-E999"


def test_event_after_outcome():
    rec = _recorder("TRC-ORDER")
    rec.record_task_received()
    rec.record_outcome("OK", success=True)
    inv = rec.record_tool_invocation("BALANCE", "get_balance", {})
    rec.record_tool_result("BALANCE", "get_balance", {"success": True},
                           related_event_id=inv.event_id)
    anomalies = DETECTOR.detect(rec.finish())
    assert _types(anomalies) == [AnomalyType.UNEXPECTED_EVENT_ORDER] * 2  # inv + result
    assert all(a.severity.value == "MEDIUM" for a in anomalies)


def test_conflicting_tool_results():
    rec = _recorder("TRC-CONF")
    rec.record_task_received()
    inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                     {"beneficiary_id": "BOB-001", "amount_inr": 10000})
    first = rec.record_tool_result("TRANSFER", "transfer",
                                   {"success": True, "reason_code": "OK"},
                                   success=True, related_event_id=inv.event_id)
    rec.record_tool_result("TRANSFER", "transfer",
                           {"success": False, "reason_code": "LIMIT_EXCEEDED"},
                           success=False, related_event_id=inv.event_id,
                           conflict_with_event_id=first.event_id)
    rec.record_outcome("LIMIT_EXCEEDED", success=False)
    anomalies = DETECTOR.detect(rec.finish())
    assert _types(anomalies) == [AnomalyType.CONFLICTING_TOOL_RESULTS]
    assert anomalies[0].severity.value == "HIGH"
    assert len(anomalies[0].event_ids) == 3  # invocation + both results


def test_action_inconsistent_with_observed_state():
    rec = _recorder("TRC-ACTION",
                    task="Transfer ₹5,000 to beneficiary BOB-002.")
    rec.record_task_received()
    check = rec.record_tool_invocation("BENEFICIARY_CHECK", "check_beneficiary",
                                       {"beneficiary_id": "BOB-002"})
    rec.record_tool_result("BENEFICIARY_CHECK", "check_beneficiary",
                           {"success": True, "reason_code": "OK", "verified": False,
                            "beneficiary_id": "BOB-002"},
                           success=True, related_event_id=check.event_id)
    inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                     {"beneficiary_id": "BOB-002", "amount_inr": 5000})
    rec.record_tool_result("TRANSFER", "transfer",
                           {"success": False, "reason_code": "BENEFICIARY_UNVERIFIED"},
                           success=False, related_event_id=inv.event_id)
    rec.record_outcome("BENEFICIARY_UNVERIFIED", success=False)
    anomalies = DETECTOR.detect(rec.finish())
    assert AnomalyType.ACTION_INCONSISTENT_WITH_STATE in _types(anomalies)


def test_llm_style_trace_with_null_state_is_not_flagged_for_state():
    """Missing state is not evidence of anomaly: no state checks, no flags."""
    rec = _recorder("TRC-NULL")
    rec.record_task_received()
    inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                     {"beneficiary_id": "BOB-001", "amount_inr": 10000},
                                     state_before=None)
    rec.record_tool_result("TRANSFER", "transfer",
                           {"success": True, "reason_code": "OK"},
                           state_after=None, success=True, related_event_id=inv.event_id)
    rec.record_outcome("OK", success=True)
    assert DETECTOR.detect(rec.finish()) == []


def test_anomaly_ids_and_determinism():
    def build_and_detect() -> str:
        rec = _recorder("TRC-DET")
        rec.record_task_received()
        for _ in range(3):
            inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                             {"beneficiary_id": "BOB-001", "amount_inr": 10})
            rec.record_tool_result("TRANSFER", "transfer",
                                   {"success": False, "reason_code": "INTERNAL_ERROR"},
                                   success=False, related_event_id=inv.event_id)
        rec.record_outcome("INTERNAL_ERROR", success=False)
        trace: ExecutionTrace = rec.finish()
        return "[ " + ", ".join(a.model_dump_json() for a in DETECTOR.detect(trace)) + " ]"

    first, second = build_and_detect(), build_and_detect()
    assert first == second
    assert '"ANOM-0001"' in first
