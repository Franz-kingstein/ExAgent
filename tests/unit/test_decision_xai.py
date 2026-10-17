"""Decision-explainability tests (Q1: requirement-to-action matching).

Covers the Phase 3 acceptance cases: all satisfied, one violated, multiple
violated, missing evidence (-> UNKNOWN, never assumed satisfied), conflicting
evidence (-> reported, never silently resolved) — each checked against known
ground truth, plus an evidence-audit test that every cited evidence item
resolves to the exact value in the trace.
"""

import pytest

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.core.trace import ExecutionTrace
from trace_xai.explain import (
    DecisionExplainer,
    RequirementStatus,
    explain_decision,
)

SAT = RequirementStatus.SATISFIED
VIOL = RequirementStatus.VIOLATED
UNK = RequirementStatus.UNKNOWN


@pytest.fixture()
def agent() -> RuleBasedAgent:
    return RuleBasedAgent(BankEnvironment.from_dataset())


@pytest.fixture()
def explainer() -> DecisionExplainer:
    return DecisionExplainer()


def _trace_for(agent, task) -> ExecutionTrace:
    return agent.run_traced(task)[1]


def _statuses(explanation) -> dict:
    return {a.requirement_id: a.status for a in explanation.requirements}


# ------------------------------------------------------------ ground truth
# Known expected statuses per scenario (evaluated against explainer output).
GROUND_TRUTH = {
    # task: {requirement: status}
    "transfer_ok": {"REQ-AMT-001": SAT, "REQ-BEN-001": SAT, "REQ-BAL-001": SAT},
    "one_violated": {"REQ-AMT-001": SAT, "REQ-BEN-001": VIOL, "REQ-BAL-001": SAT},
    "multi_violated": {"REQ-AMT-001": VIOL, "REQ-BEN-001": VIOL, "REQ-BAL-001": SAT},
}


def test_case1_all_requirements_satisfied(agent, explainer):
    trace = _trace_for(agent, "Transfer ₹10,000 to beneficiary BOB-001.")
    explanation = explainer.explain(trace)
    assert _statuses(explanation) == GROUND_TRUTH["transfer_ok"]
    assert all(a.confidence == 0.95 for a in explanation.requirements)
    assert all(a.conflicting_evidence == [] for a in explanation.requirements)
    assert explanation.llm_used is False
    assert "within the per-transaction limit" in explanation.rationale
    assert "verified according to the recorded verification result" in explanation.rationale
    assert "sufficient" in explanation.rationale


def test_case2_one_requirement_violated(agent, explainer):
    trace = _trace_for(agent, "Transfer ₹5,000 to beneficiary BOB-002.")
    explanation = explainer.explain(trace)
    assert _statuses(explanation) == GROUND_TRUTH["one_violated"]
    violated = next(a for a in explanation.requirements if a.requirement_id == "REQ-BEN-001")
    assert violated.status is VIOL
    # both deterministic state evidence and the environment's reason agree
    fields = {e.field for e in violated.supporting_evidence}
    assert "state_before.beneficiaries_verified.BOB-002" in fields
    assert "tool_output.reason_code" in fields
    assert "not verified" in explanation.rationale


def test_case3_multiple_requirements_violated(agent, explainer):
    trace = _trace_for(agent, "Transfer ₹60,000 to beneficiary BOB-002.")
    explanation = explainer.explain(trace)
    assert _statuses(explanation) == GROUND_TRUTH["multi_violated"]
    # environment rejects with first-fail BENEFICIARY_UNVERIFIED, yet the
    # trace's state evidence still reveals the second violation (amount over
    # limit) — multiple violations detected from one trace
    amounts = next(a for a in explanation.requirements if a.requirement_id == "REQ-AMT-001")
    assert "exceeded" not in amounts.explanation and "above" in amounts.explanation


def test_case4a_missing_evidence_entirely(agent, explainer):
    trace = _trace_for(agent, "Order a pizza with extra cheese")  # no tools ran
    explanation = explainer.explain(trace)
    assert explanation.requirements == []  # no applicable action observed
    assert "No registered requirements apply" in explanation.rationale


def test_case4b_missing_state_is_unknown_not_satisfied(explainer):
    """Trace without state: only the outcome can resolve some requirements."""
    rec = ExecutionRecorder(
        task_id="TASK-MISS-0001", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="manual-agent",
    )
    rec.record_task_received()
    inv = rec.record_tool_invocation(
        "TRANSFER", "transfer", {"beneficiary_id": "BOB-001", "amount_inr": 10000},
        state_before=None,  # state unavailable
    )
    rec.record_tool_result(
        "TRANSFER", "transfer",
        {"success": False, "reason_code": "INSUFFICIENT_FUNDS", "beneficiary_id": "BOB-001",
         "amount_inr": 10000},
        state_after=None, success=False, related_event_id=inv.event_id,
    )
    rec.record_outcome("INSUFFICIENT_FUNDS", success=False)
    explanation = explainer.explain(rec.finish())

    assert _statuses(explanation) == {
        "REQ-AMT-001": UNK, "REQ-BEN-001": UNK, "REQ-BAL-001": VIOL,
    }
    for assessment in explanation.requirements:
        if assessment.status is UNK:
            assert assessment.confidence == 0.0
            assert assessment.supporting_evidence == []
            assert "No observable evidence" in assessment.explanation


def test_case5a_conflicting_amount_evidence(explainer):
    """One invocation with two contradicting results -> UNKNOWN + conflict."""
    rec = ExecutionRecorder(
        task_id="TASK-CONF-0001", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="manual-agent",
    )
    rec.record_task_received()
    inv = rec.record_tool_invocation(
        "TRANSFER", "transfer", {"beneficiary_id": "BOB-001", "amount_inr": 10000},
        state_before={
            "balances": {"ACC-SYN-001": 250000},
            "per_txn_limits": {"ACC-SYN-001": 50000},
            "beneficiaries_verified": {"BOB-001": True},
        },
    )
    first = rec.record_tool_result(
        "TRANSFER", "transfer",
        {"success": True, "reason_code": "OK", "account_id": "ACC-SYN-001"},
        success=True, related_event_id=inv.event_id,
    )
    rec.record_tool_result(
        "TRANSFER", "transfer", {"success": False, "reason_code": "LIMIT_EXCEEDED"},
        success=False, related_event_id=inv.event_id,
        conflict_with_event_id=first.event_id,
    )
    rec.record_outcome("LIMIT_EXCEEDED", success=False)
    explanation = explainer.explain(rec.finish())

    amount = next(a for a in explanation.requirements if a.requirement_id == "REQ-AMT-001")
    assert amount.status is UNK
    assert amount.confidence == 0.0
    assert amount.supporting_evidence == []
    assert len(amount.conflicting_evidence) >= 3  # state comparison + both reasons
    assert "Conflicting evidence" in amount.explanation
    # other requirements remain resolvable
    statuses = _statuses(explanation)
    assert statuses["REQ-BEN-001"] is SAT and statuses["REQ-BAL-001"] is SAT


def test_case5b_conflicting_verification_evidence(explainer):
    """State says verified, a check tool says otherwise -> report conflict."""
    rec = ExecutionRecorder(
        task_id="TASK-CONF-0002", task_text="Transfer ₹1,000 to beneficiary BOB-001.",
        agent_id="manual-agent",
    )
    rec.record_task_received()
    inv = rec.record_tool_invocation(
        "TRANSFER", "transfer", {"beneficiary_id": "BOB-001", "amount_inr": 1000},
        state_before={
            "balances": {"ACC-SYN-001": 250000},
            "per_txn_limits": {"ACC-SYN-001": 50000},
            "beneficiaries_verified": {"BOB-001": True},
        },
    )
    check = rec.record_tool_invocation("BENEFICIARY_CHECK", "check_beneficiary",
                                       {"beneficiary_id": "BOB-001"})
    rec.record_tool_result(
        "BENEFICIARY_CHECK", "check_beneficiary",
        {"success": True, "reason_code": "OK", "verified": False},
        success=True, related_event_id=check.event_id,
    )
    rec.record_tool_result(
        "TRANSFER", "transfer", {"success": True, "reason_code": "OK",
                                 "account_id": "ACC-SYN-001"},
        success=True, related_event_id=inv.event_id,
    )
    rec.record_outcome("OK", success=True)
    explanation = explainer.explain(rec.finish())

    verified = next(a for a in explanation.requirements if a.requirement_id == "REQ-BEN-001")
    assert verified.status is UNK  # deterministic state vs tool result disagree
    conflict_fields = {e.field for e in verified.conflicting_evidence}
    assert "state_before.beneficiaries_verified.BOB-001" in conflict_fields
    assert "tool_output.verified" in conflict_fields
    assert verified.conflicting_evidence and verified.supporting_evidence == []


def test_ground_truth_matrix(agent, explainer):
    """Explicit evaluation of explainer output vs known ground truth."""
    scenarios = {
        "transfer_ok": "Transfer ₹10,000 to beneficiary BOB-001.",
        "one_violated": "Transfer ₹5,000 to beneficiary BOB-002.",
        "multi_violated": "Transfer ₹60,000 to beneficiary BOB-002.",
    }
    for name, task in scenarios.items():
        explanation = explainer.explain(_trace_for(agent, task))
        assert _statuses(explanation) == GROUND_TRUTH[name], f"scenario {name}"


def test_non_transfer_action_has_no_applicable_requirements(agent, explainer):
    trace = _trace_for(agent, "What is the balance of account ACC-SYN-001?")
    explanation = explainer.explain(trace)
    assert explanation.action == "BALANCE"
    assert explanation.requirements == []


def test_every_cited_evidence_resolves_in_the_trace(agent, explainer):
    """Auditability: each evidence item must point to the exact trace value."""
    trace = _trace_for(agent, "Transfer ₹10,000 to beneficiary BOB-001.")
    explanation = explainer.explain(trace)
    events = {e.event_id: e.model_dump(mode="json") for e in trace.events}

    def resolve(event_dump, path: str):
        node = event_dump
        for part in path.split("."):
            node = node[part]
        return node

    checked = 0
    for assessment in explanation.requirements:
        for item in assessment.supporting_evidence + assessment.conflicting_evidence:
            assert item.event_id in events, f"dangling evidence ref {item.event_id}"
            assert resolve(events[item.event_id], item.field) == item.value, (
                f"evidence {item.event_id}.{item.field} does not match the trace"
            )
            checked += 1
    assert checked >= 6  # every requirement cited at least one item


def test_convenience_api_and_determinism(agent):
    t1 = _trace_for(agent, "Transfer ₹10,000 to beneficiary BOB-001.")
    e1 = explain_decision(t1)
    e2 = explain_decision(t1)
    assert e1 == e2
    assert e1.method == "requirement_to_action_matching_v1"
    assert e1.llm_used is False
