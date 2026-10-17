"""Hybrid deviation-explainability tests (Q2) against known simulated ground truth.

Scenarios: normal, requirement violation without anomaly, anomaly without
violation, both, missing trace event, conflicting trace information.
"""

import re

import pytest

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.explain import (
    AnomalyType,
    DeviationCategory,
    DeviationExplainer,
    RelationLabel,
    explain_deviation,
)

STATE_OK = {
    "balances": {"ACC-SYN-001": 250000},
    "per_txn_limits": {"ACC-SYN-001": 50000},
    "beneficiaries_verified": {"BOB-001": True},
}

@pytest.fixture()
def explainer() -> DeviationExplainer:
    return DeviationExplainer()


def _recorder(trace_id, task) -> ExecutionRecorder:
    return ExecutionRecorder(
        task_id=f"TASK-{trace_id}", task_text=task, agent_id="manual-agent", trace_id=trace_id
    )


def _retry_then_success_trace():
    """2 failed attempts, then success: anomaly, but all requirements hold."""
    rec = _recorder("TRC-S3", "Transfer ₹10,000 to beneficiary BOB-001.")
    rec.record_task_received()
    for _ in range(2):
        inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                         {"beneficiary_id": "BOB-001", "amount_inr": 10000},
                                         state_before=STATE_OK)
        rec.record_tool_result("TRANSFER", "transfer",
                               {"success": False, "reason_code": "INTERNAL_ERROR"},
                               state_after=STATE_OK, success=False,
                               related_event_id=inv.event_id)
    inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                     {"beneficiary_id": "BOB-001", "amount_inr": 10000},
                                     state_before=STATE_OK)
    rec.record_tool_result(
        "TRANSFER", "transfer",
        {"success": True, "reason_code": "OK", "account_id": "ACC-SYN-001",
         "amount_inr": 10000, "balance_inr": 240000},
        state_after={**STATE_OK, "balances": {"ACC-SYN-001": 240000}},
        success=True, related_event_id=inv.event_id,
    )
    rec.record_outcome("OK", success=True)
    return rec.finish()


def _over_limit_retry_trace():
    """3 attempts of an over-limit transfer: violation + excessive retry."""
    rec = _recorder("TRC-S4", "Transfer ₹60,000 to beneficiary BOB-001.")
    rec.record_task_received()
    for _ in range(3):
        inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                         {"beneficiary_id": "BOB-001", "amount_inr": 60000},
                                         state_before=STATE_OK)
        rec.record_tool_result("TRANSFER", "transfer",
                               {"success": False, "reason_code": "LIMIT_EXCEEDED"},
                               state_after=STATE_OK, success=False,
                               related_event_id=inv.event_id)
    rec.record_outcome("LIMIT_EXCEEDED", success=False)
    return rec.finish()


def _missing_event_trace():
    rec = _recorder("TRC-S5", "Transfer ₹10,000 to beneficiary BOB-001.")
    rec.record_task_received()
    rec.record_tool_result("TRANSFER", "transfer",
                           {"success": True, "reason_code": "OK"},
                           success=True, related_event_id="TRC-S5-E999")  # deleted event
    rec.record_outcome("OK", success=True)
    return rec.finish()


def _conflicting_trace():
    rec = _recorder("TRC-S6", "Transfer ₹10,000 to beneficiary BOB-001.")
    rec.record_task_received()
    inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                     {"beneficiary_id": "BOB-001", "amount_inr": 10000},
                                     state_before=STATE_OK)
    first = rec.record_tool_result(
        "TRANSFER", "transfer",
        {"success": True, "reason_code": "OK", "account_id": "ACC-SYN-001"},
        state_after={**STATE_OK, "balances": {"ACC-SYN-001": 240000}},
        success=True, related_event_id=inv.event_id,
    )
    rec.record_tool_result("TRANSFER", "transfer",
                           {"success": False, "reason_code": "LIMIT_EXCEEDED"},
                           success=False, related_event_id=inv.event_id,
                           conflict_with_event_id=first.event_id)
    rec.record_outcome("LIMIT_EXCEEDED", success=False)
    return rec.finish()


# ------------------------------------------------------------- scenarios
def test_scenario1_normal_trajectory(explainer):
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")[1]
    deviation = explainer.explain(trace)
    assert deviation.category is DeviationCategory.NONE
    assert deviation.requirement_violations == []
    assert deviation.anomalies == []
    assert deviation.relation is RelationLabel.NONE
    assert "No deviation detected" in deviation.combined_explanation
    assert deviation.llm_used is False


def test_scenario2_violation_without_anomaly(explainer):
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    trace = agent.run_traced("Transfer ₹60,000 to beneficiary BOB-001.")[1]
    deviation = explainer.explain(trace)
    assert deviation.category is DeviationCategory.REQUIREMENT_VIOLATION
    assert [v.requirement_id for v in deviation.requirement_violations] == ["REQ-AMT-001"]
    assert deviation.anomalies == []
    assert deviation.relation is RelationLabel.NONE
    assert "above the per-transaction limit" in deviation.combined_explanation


def test_scenario3_anomaly_without_violation(explainer):
    deviation = explainer.explain(_retry_then_success_trace())
    assert deviation.category is DeviationCategory.TRAJECTORY_ANOMALY
    assert deviation.requirement_violations == []
    assert [a.anomaly_type for a in deviation.anomalies] == [
        AnomalyType.REPEATED_FAILED_ACTION]
    assert deviation.relation is RelationLabel.NONE
    assert "failed 2 time(s)" in deviation.combined_explanation


def test_scenario4_both_anomaly_and_violation(explainer):
    deviation = explainer.explain(_over_limit_retry_trace())
    assert deviation.category is DeviationCategory.BOTH
    assert [v.requirement_id for v in deviation.requirement_violations] == ["REQ-AMT-001"]
    assert [a.anomaly_type for a in deviation.anomalies] == [AnomalyType.EXCESSIVE_RETRY]
    assert deviation.relation is RelationLabel.CONFIRMED  # same tool calls
    combined = deviation.combined_explanation
    assert "was VIOLATED" in combined
    assert "The execution trace also contains" in combined
    assert "No causal relationship is claimed" in combined
    # the spec's hard rule: never assert causation
    assert "caused" not in combined.lower() and "because of" not in combined.lower()


def test_scenario5_missing_trace_event(explainer):
    deviation = explainer.explain(_missing_event_trace())
    assert deviation.category is DeviationCategory.TRAJECTORY_ANOMALY
    assert [a.anomaly_type for a in deviation.anomalies] == [
        AnomalyType.UNEXPECTED_EVENT_ORDER]
    assert deviation.requirement_violations == []
    assert "no resolvable invocation" in deviation.combined_explanation


def test_scenario6_conflicting_trace_information(explainer):
    deviation = explainer.explain(_conflicting_trace())
    assert deviation.category is DeviationCategory.TRAJECTORY_ANOMALY
    assert [a.anomaly_type for a in deviation.anomalies] == [
        AnomalyType.CONFLICTING_TOOL_RESULTS]
    assert deviation.requirement_violations == []  # conflicting req -> UNKNOWN, not picked
    assert deviation.unknown_requirement_ids == ["REQ-AMT-001"]
    assert deviation.relation is RelationLabel.UNKNOWN
    assert "evidence was insufficient for REQ-AMT-001" in deviation.combined_explanation
    assert "unknown due to insufficient evidence" in deviation.combined_explanation


# ---------------------------------------------------------- ground truth
GROUND_TRUTH = {
    "normal": (DeviationCategory.NONE, [], []),
    "violation_only": (DeviationCategory.REQUIREMENT_VIOLATION, ["REQ-AMT-001"], []),
    "anomaly_only": (DeviationCategory.TRAJECTORY_ANOMALY, [],
                     [AnomalyType.REPEATED_FAILED_ACTION]),
    "both": (DeviationCategory.BOTH, ["REQ-AMT-001"], [AnomalyType.EXCESSIVE_RETRY]),
    "missing_event": (DeviationCategory.TRAJECTORY_ANOMALY, [],
                      [AnomalyType.UNEXPECTED_EVENT_ORDER]),
    "conflicting": (DeviationCategory.TRAJECTORY_ANOMALY, [],
                    [AnomalyType.CONFLICTING_TOOL_RESULTS]),
}


def test_ground_truth_matrix(explainer):
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    scenarios = {
        "normal": agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")[1],
        "violation_only": agent.run_traced("Transfer ₹60,000 to beneficiary BOB-001.")[1],
        "anomaly_only": _retry_then_success_trace(),
        "both": _over_limit_retry_trace(),
        "missing_event": _missing_event_trace(),
        "conflicting": _conflicting_trace(),
    }
    for name, trace in scenarios.items():
        deviation = explainer.explain(trace)
        category, violation_ids, anomaly_types = GROUND_TRUTH[name]
        assert deviation.category is category, f"category mismatch for {name}"
        assert [v.requirement_id for v in deviation.requirement_violations] == violation_ids
        assert [a.anomaly_type for a in deviation.anomalies] == anomaly_types


def test_anomaly_output_shape(explainer):
    deviation = explainer.explain(_over_limit_retry_trace())
    for anomaly in deviation.anomalies:
        assert re.match(r"^ANOM-\d{4}$", anomaly.anomaly_id)
        assert anomaly.event_ids
        assert 0.0 <= anomaly.confidence <= 1.0
        assert anomaly.evidence or anomaly.anomaly_type.value == "MISSING_EXPECTED_EVENT"


def test_determinism(explainer):
    trace = _over_limit_retry_trace()
    assert explainer.explain(trace) == explainer.explain(trace)


def test_convenience_api(explainer):
    assert explain_deviation(_retry_then_success_trace()).category is (
        DeviationCategory.TRAJECTORY_ANOMALY
    )
