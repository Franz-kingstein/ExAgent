"""Phase 5 (Q3) tests: counterfactual explainability.

All fixtures use the synthetic BankEnvironment + deterministic rule agent.
The required scenarios are covered:

1. Alternative satisfies requirement
2. Alternative still fails requirement
3. Alternative improves outcome
4. Alternative worsens outcome
5. Replay unavailable
6. Evidence insufficient
"""

import pytest

from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.core.trace import ExecutionTrace
from trace_xai.explain.counterfactual import CounterfactualExplainer
from trace_xai.explain.models import (
    ActionSpec,
    CounterfactualImpact,
    RequirementStatus,
)


# ------------------------------------------------------------- scenario 1 & 3
class TestAlternativeSatisfiesRequirement:
    """Actual INR 75,000 > limit 50,000 -> generated INR 50,000 succeeds."""

    def test_minimal_alternative_satisfies_requirement(self):
        task = "Transfer ₹75,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        explainer = CounterfactualExplainer()
        explanation = explainer.explain(trace)

        # actual action failed with LIMIT_EXCEEDED
        assert "rejected" in explanation.actual_summary

        # exactly one minimal candidate (clamp to limit; balance clamp = same
        # amount is deduplicated)
        assert len(explanation.counterfactuals) == 1
        cf = explanation.counterfactuals[0]
        assert cf.alternative_action.amount_inr == 50_000
        assert cf.original_action.amount_inr == 75_000
        assert cf.alternative_action.beneficiary_id == "BOB-001"
        assert "per-transaction limit 50000" in cf.changed_condition

        # predicted success with the documented requirement flip
        assert cf.predicted_outcome.success is True
        assert cf.predicted_outcome.reason_code == "OK"
        comparison = {c.requirement_id: c for c in cf.requirement_comparison}
        assert comparison["REQ-AMT-001"].actual_status is RequirementStatus.VIOLATED
        assert comparison["REQ-AMT-001"].alternative_status is RequirementStatus.SATISFIED
        assert comparison["REQ-AMT-001"].changed is True
        assert comparison["REQ-BEN-001"].actual_status is RequirementStatus.SATISFIED
        assert comparison["REQ-BAL-001"].actual_status is RequirementStatus.SATISFIED
        assert cf.predicted_outcome.expected_state_change == "ACC-SYN-001 balance 250000 -> 200000"

        # controlled replay actually executed it and confirmed the prediction
        assert cf.replay is not None and cf.replay.performed is True
        assert cf.replay.execution_success is True
        assert cf.replay.transaction_id is not None
        assert cf.replay.state_change["balances_after"]["ACC-SYN-001"] == 200_000

        # scenario 3: outcome improves (failed -> would succeed, replay-confirmed)
        assert cf.impact is CounterfactualImpact.IMPROVES
        assert "Controlled replay confirmed" in cf.explanation
        assert explanation.llm_used is False

    def test_counterfactual_is_deterministic(self):
        task = "Transfer ₹75,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)
        first = CounterfactualExplainer().explain(trace)
        second = CounterfactualExplainer().explain(trace)
        assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------- scenario 2
class TestAlternativeStillFails:
    """ACC-SYN-002: limit 10,000, balance 5,000, actual INR 20,000."""

    def test_clamp_to_limit_still_fails_on_balance(self):
        task = "Transfer ₹20,000 to beneficiary BOB-001 from account ACC-SYN-002."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        explanation = CounterfactualExplainer().explain(trace)
        by_amount = {
            cf.alternative_action.amount_inr: cf
            for cf in explanation.counterfactuals
        }
        assert set(by_amount) == {10_000, 5_000}  # limit clamp + balance clamp

        # clamp to limit only: still fails, but for a *different* reason
        still_fails = by_amount[10_000]
        assert still_fails.predicted_outcome.success is False
        assert still_fails.predicted_outcome.reason_code == "INSUFFICIENT_FUNDS"
        comparison = {c.requirement_id: c for c in still_fails.requirement_comparison}
        assert comparison["REQ-AMT-001"].alternative_status is RequirementStatus.SATISFIED
        assert comparison["REQ-BAL-001"].alternative_status is RequirementStatus.VIOLATED
        assert still_fails.impact is CounterfactualImpact.UNCHANGED
        assert still_fails.replay is not None
        assert still_fails.replay.execution_success is False
        assert still_fails.replay.reason_code == "INSUFFICIENT_FUNDS"
        assert "still have been rejected" in still_fails.explanation
        assert "REQ-BAL-001 remain VIOLATED" in still_fails.explanation

        # clamp to binding constraint min(limit, balance): succeeds
        improves = by_amount[5_000]
        assert improves.predicted_outcome.success is True
        assert improves.impact is CounterfactualImpact.IMPROVES
        assert improves.replay is not None and improves.replay.execution_success is True


# --------------------------------------------------------------- scenario 4
class TestAlternativeWorsensOutcome:
    def test_provided_larger_amount_worsens_success(self):
        task = "Transfer ₹10,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)
        assert trace.final_outcome[0] is True  # actual succeeded

        explainer = CounterfactualExplainer()
        explanation = explainer.explain(
            trace, alternatives=[ActionSpec(amount_inr=60_000, beneficiary_id="BOB-001")]
        )

        # nothing minimal to improve a clean success
        assert all(cf.source == "provided" for cf in explanation.counterfactuals)
        assert any("nothing to improve" in n for n in explanation.notes)

        cf = explanation.counterfactuals[0]
        assert cf.source == "provided"
        assert cf.changed_condition == "user-provided alternative action"
        assert cf.predicted_outcome.success is False
        assert cf.predicted_outcome.reason_code == "LIMIT_EXCEEDED"
        assert cf.impact is CounterfactualImpact.WORSENS
        assert cf.replay is not None and cf.replay.execution_success is False
        assert cf.replay.reason_code == "LIMIT_EXCEEDED"
        comparison = {c.requirement_id: c for c in cf.requirement_comparison}
        assert comparison["REQ-AMT-001"].alternative_status is RequirementStatus.VIOLATED
        assert comparison["REQ-AMT-001"].changed is True


# --------------------------------------------------------------- scenario 5
class TestReplayUnavailable:
    def test_replay_disabled_still_predicts_but_claims_no_execution(self):
        task = "Transfer ₹75,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        explainer = CounterfactualExplainer(replay_enabled=False)
        explanation = explainer.explain(trace)

        cf = explanation.counterfactuals[0]
        assert cf.replay is None  # never claims execution
        assert cf.predicted_outcome.success is True  # prediction still produced
        assert "Not replayed: prediction only" in cf.explanation
        # uncertainty reflects the missing validation (+0.10 over replayed run)
        assert cf.uncertainty == pytest.approx(0.15)
        assert any("replay disabled" in n for n in explanation.notes)

    def test_replay_enabled_has_lower_uncertainty_than_disabled(self):
        task = "Transfer ₹75,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)
        replayed = CounterfactualExplainer(replay_enabled=True).explain(trace)
        simulated = CounterfactualExplainer(replay_enabled=False).explain(trace)
        assert replayed.counterfactuals[0].uncertainty < simulated.counterfactuals[0].uncertainty


# --------------------------------------------------------------- scenario 6
def _stateless_trace() -> "ExecutionTrace":
    """An LLM-style trace without recorded pre-action state."""
    recorder = ExecutionRecorder(
        task_id="task-stateless",
        task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b",
    )
    recorder.record_task_received()
    recorder.record_tool_invocation(
        action="transfer",
        tool_name="transfer",
        tool_input={"amount_inr": 10_000, "beneficiary_id": "BOB-001"},
        state_before=None,  # no state evidence recorded
    )
    recorder.record_tool_result(
        action="transfer",
        tool_name="transfer",
        tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS"},
        success=False,
    )
    recorder.record_outcome("INSUFFICIENT_FUNDS", success=False)
    return recorder.finish()


class TestEvidenceInsufficient:
    def test_stateless_trace_yields_no_minimal_alternatives(self):
        trace = _stateless_trace()

        explainer = CounterfactualExplainer()
        explanation = explainer.explain(trace)

        # without state evidence no minimal alternative can be generated
        assert explanation.counterfactuals == []
        assert any("No recorded pre-action state" in n for n in explanation.notes)

    def test_provided_alternative_on_stateless_trace_is_fully_unknown(self):
        trace = _stateless_trace()

        explainer = CounterfactualExplainer()
        explanation = explainer.explain(
            trace, alternatives=[ActionSpec(amount_inr=5_000, beneficiary_id="BOB-001")]
        )

        assert len(explanation.counterfactuals) == 1
        cf = explanation.counterfactuals[0]
        # everything UNKNOWN, nothing fabricated, no execution claim
        assert cf.predicted_outcome.success is None
        assert all(
            s is RequirementStatus.UNKNOWN
            for s in cf.predicted_outcome.requirement_statuses.values()
        )
        assert cf.impact is CounterfactualImpact.UNKNOWN
        assert cf.replay is None
        assert cf.uncertainty >= 0.5
        assert "cannot be determined" in cf.explanation


# ----------------------------------------------------------- generation rules
class TestMinimalChangeGeneration:
    def test_beneficiary_switch_only_when_unverified(self):
        # actual: BOB-002 is unverified (linked); amount is fine
        task = "Transfer ₹5,000 to beneficiary BOB-002."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)
        assert trace.final_outcome[1] == "BENEFICIARY_UNVERIFIED"

        explanation = CounterfactualExplainer().explain(trace)
        assert len(explanation.counterfactuals) == 1
        cf = explanation.counterfactuals[0]
        # minimal edit: switch payee, keep the amount
        assert cf.alternative_action.amount_inr == 5_000
        assert cf.alternative_action.beneficiary_id == "BOB-001"
        assert "switched from unverified BOB-002 to verified BOB-001" in cf.changed_condition
        assert cf.predicted_outcome.success is True
        assert cf.impact is CounterfactualImpact.IMPROVES
        comparison = {c.requirement_id: c for c in cf.requirement_comparison}
        assert comparison["REQ-BEN-001"].changed is True

    def test_no_alternatives_when_nothing_to_improve(self):
        task = "Transfer ₹10,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)
        assert trace.final_outcome[0] is True

        explanation = CounterfactualExplainer().explain(trace)
        # a clean minimal success generates no fabricated alternatives
        assert explanation.counterfactuals == []
        assert any("nothing to improve" in n for n in explanation.notes)

    def test_provided_alternative_equal_to_original_is_skipped(self):
        task = "Transfer ₹10,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        explanation = CounterfactualExplainer().explain(
            trace,
            alternatives=[ActionSpec(amount_inr=10_000, beneficiary_id="BOB-001")],
        )
        assert explanation.counterfactuals == []
        assert any("equals the original action" in n for n in explanation.notes)


# ------------------------------------------------------------------- honesty
class TestSimulationVsReplayHonesty:
    def test_simulation_never_claims_execution(self):
        task = "Transfer ₹75,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        # replay disabled: the counterfactual must never read as executed
        explanation = CounterfactualExplainer(replay_enabled=False).explain(trace)
        cf = explanation.counterfactuals[0]
        assert cf.replay is None
        assert cf.predicted_outcome.basis == "deterministic_simulation"
        assert "execution claim" not in cf.predicted_outcome.expected_state_change

    def test_replay_record_is_explicit_when_performed(self):
        task = "Transfer ₹75,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        explanation = CounterfactualExplainer().explain(trace)
        cf = explanation.counterfactuals[0]
        assert cf.replay is not None
        assert "isolated clone of the synthetic environment" in cf.replay.note
        # the replayed transaction id is recorded on the replay record only
        assert cf.replay.transaction_id == "TXN-SYN-0001"

    def test_evidence_cites_trace_fields_only(self):
        task = "Transfer ₹75,000 to beneficiary BOB-001."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        explanation = CounterfactualExplainer().explain(trace)
        cf = explanation.counterfactuals[0]
        event_ids = {e.event_id for e in trace.events}
        assert cf.evidence
        for item in cf.evidence:
            assert item.event_id in event_ids
            assert "." in item.field

    def test_counterfactual_ids_are_ordered_and_well_formed(self):
        task = "Transfer ₹20,000 to beneficiary BOB-001 from account ACC-SYN-002."
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced(task)

        explanation = CounterfactualExplainer().explain(trace)
        ids = [cf.cf_id for cf in explanation.counterfactuals]
        assert ids == [f"CF-{i:04d}" for i in range(1, len(ids) + 1)]
