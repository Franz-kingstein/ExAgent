"""End-to-end tests: the integrated TRACE-XAI pipeline (scenarios A-I).

A. Successful banking action      F. Missing evidence
B. Requirement violation          G. Conflicting evidence
C. Trajectory anomaly             H. Unsupported explanation
D. Violation + anomaly            I. Correct explanation
E. Counterfactual improvement

Every test drives the full pipeline (trace -> Q1/Q2/Q3 -> rationale ->
validation -> FinalXAIReport) and asserts on the 15-section report.
"""

import json

import pytest

from trace_xai import FinalXAIReport, TraceXAIPipeline, analyze_execution
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.explain.models import (
    DeviationCategory,
    RelationLabel,
    RequirementStatus,
)
from trace_xai.validate.models import (
    GuardrailDecision,
    OverallValidationStatus,
    VerificationStatus,
)


@pytest.fixture
def pipeline() -> TraceXAIPipeline:
    return TraceXAIPipeline(agent=RuleBasedAgent(BankEnvironment.from_dataset()))


def _report_shape(report: FinalXAIReport):
    """The 15 sections must all exist and be populated with real structure."""
    assert report.task
    assert report.requirements
    assert report.agent_action
    assert report.decision_rationale
    assert report.requirement_satisfaction
    assert report.deviations is not None
    assert report.trajectory_anomalies is not None
    assert report.counterfactuals is not None
    assert report.evidence
    assert report.validation.verifications is not None
    assert report.supported_claims is not None
    assert report.contradicted_claims is not None
    assert report.unknown_claims is not None
    assert report.uncertainty
    assert report.limitations
    text = report.to_text()
    for i, title in enumerate(
        ["Task", "Requirements", "Agent Action", "Decision Rationale",
         "Requirement Satisfaction", "Deviations", "Trajectory Anomalies",
         "Counterfactual Alternatives", "Evidence", "Validation Results",
         "Supported Claims", "Contradicted Claims", "Unknown Claims",
         "Confidence / Uncertainty", "Limitations"], start=1):
        assert f"[{i}] {title}" in text
    return text


# ------------------------------------------------------------- A. success
class TestScenarioA_SuccessfulAction:
    def test_success_report(self, pipeline):
        _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        _report_shape(report)

        assert all(s == RequirementStatus.SATISFIED.value
                   for s in report.requirement_satisfaction.values())
        assert report.deviation.category is DeviationCategory.NONE
        assert report.validation.overall_validation_status \
            is OverallValidationStatus.SUPPORTED
        assert report.validation.guardrail_decision is GuardrailDecision.PASS
        assert report.contradicted_claims == []
        assert report.timings_ms and set(report.timings_ms) == {
            "decision", "deviation", "counterfactual", "rationale", "validation"}

    def test_analyze_execution_by_task_id(self, pipeline):
        # the single-call API: run once, analyze by id later
        result, first = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        assert result.success
        again = pipeline.analyze_execution(task_id=first.task_id)
        assert again.trace_id == first.trace_id
        # re-analysis of the same trace is deterministic; only the wall-clock
        # measurements (stage timings, agent execution time) legitimately
        # differ between passes
        a, b = first.model_dump(), again.model_dump()
        for key in ("timings_ms", "agent_execution_ms"):
            a.pop(key), b.pop(key)
        assert a == b

    def test_one_call_helper(self):
        report = analyze_execution(
            "Transfer ₹10,000 to beneficiary BOB-001.",
            agent=RuleBasedAgent(BankEnvironment.from_dataset()))
        assert isinstance(report, FinalXAIReport)
        _report_shape(report)


# ---------------------------------------------------------- B. violation
class TestScenarioB_RequirementViolation:
    def test_violation_report(self, pipeline):
        _, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
        _report_shape(report)

        assert report.requirement_satisfaction["REQ-AMT-001"] == "VIOLATED"
        assert report.deviation.category is DeviationCategory.REQUIREMENT_VIOLATION
        # the rationale + claims stay evidence-grounded and validated
        assert "REQ-AMT-001" in report.decision_rationale
        assert report.validation.guardrail_decision is GuardrailDecision.PASS
        assert report.validation.contradicted_claim_count == 0

    def test_unknown_task_id_raises(self, pipeline):
        with pytest.raises(KeyError):
            pipeline.analyze_execution(task_id="NOPE")


# ---------------------------------------------------------- C. anomaly
def _repeated_invalid_amount_trace():
    """Two identical invalid-amount transfers -> REPEATED_FAILED_ACTION,
    no registered requirement violated."""
    env = BankEnvironment.from_dataset()
    recorder = ExecutionRecorder(
        task_id="TASK-SYN-C", task_text="Transfer an invalid amount.",
        agent_id="rule-based-agent", trace_id="TRC-SYN-C",
    )
    recorder.record_task_received()
    for _ in range(2):
        invocation = recorder.record_tool_invocation(
            action="transfer", tool_name="transfer",
            tool_input={"amount_inr": -5000, "beneficiary_id": "BOB-001"},
            state_before=env.summary(),
        )
        recorder.record_tool_result(
            action="transfer", tool_name="transfer",
            tool_output={"success": False, "reason_code": "INVALID_AMOUNT",
                         "account_id": "ACC-SYN-001"},
            success=False,
            related_event_id=invocation.event_id,
        )
    recorder.record_outcome("INVALID_AMOUNT", success=False)
    return recorder.finish()


class TestScenarioC_TrajectoryAnomaly:
    def test_anomaly_report(self, pipeline):
        report = pipeline.analyze_execution(trace=_repeated_invalid_amount_trace())
        _report_shape(report)

        assert report.deviation.category is DeviationCategory.TRAJECTORY_ANOMALY
        assert report.deviation.anomalies
        assert any(a.anomaly_type.value == "REPEATED_FAILED_ACTION"
                   for a in report.deviation.anomalies)
        # no registered requirement is violated in this trace
        assert all(s != RequirementStatus.VIOLATED.value
                   for s in report.requirement_satisfaction.values())
        # no requirement violation exists, so there is no anomaly<->violation
        # association to disclaim; the report must not fabricate one
        assert report.deviation.relation is RelationLabel.NONE


# ---------------------------------------------------- D. violation + anomaly
class TestScenarioD_ViolationAndAnomaly:
    def test_combined_report(self, pipeline):
        # 75k twice: requirement violation AND repeated-failure anomaly
        env = BankEnvironment.from_dataset()
        recorder = ExecutionRecorder(
            task_id="TASK-SYN-D", task_text="Transfer ₹75,000 to BOB-001 (twice).",
            agent_id="rule-based-agent", trace_id="TRC-SYN-D",
        )
        recorder.record_task_received()
        for _ in range(2):
            invocation = recorder.record_tool_invocation(
                action="transfer", tool_name="transfer",
                tool_input={"amount_inr": 75_000, "beneficiary_id": "BOB-001"},
                state_before=env.summary(),
            )
            recorder.record_tool_result(
                action="transfer", tool_name="transfer",
                tool_output={"success": False, "reason_code": "LIMIT_EXCEEDED",
                             "account_id": "ACC-SYN-001"},
                success=False,
                related_event_id=invocation.event_id,
            )
        recorder.record_outcome("LIMIT_EXCEEDED", success=False)

        report = pipeline.analyze_execution(trace=recorder.finish())
        _report_shape(report)

        assert report.deviation.category is DeviationCategory.BOTH
        assert report.requirement_satisfaction["REQ-AMT-001"] == "VIOLATED"
        assert report.deviation.anomalies
        assert report.deviation.relation is RelationLabel.CONFIRMED


# -------------------------------------------- E. counterfactual improvement
class TestScenarioE_CounterfactualImprovement:
    def test_improving_alternative_report(self, pipeline):
        _, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
        cfs = report.counterfactual_explanation.counterfactuals
        assert cfs, "over-limit transfer must yield a minimal alternative"
        best = cfs[0]
        assert best.alternative_action.amount_inr == 50_000
        assert best.impact.value == "improves"
        assert best.replay is not None and best.replay.execution_success
        # section 8 surfaces it to the user-facing report
        assert any("CF-0001" in line for line in report.counterfactuals)


# --------------------------------------------------------- F. missing evidence
def _stateless_trace():
    recorder = ExecutionRecorder(
        task_id="TASK-SYN-F", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b", trace_id="TRC-SYN-F",
    )
    recorder.record_task_received()
    recorder.record_tool_invocation(
        action="transfer", tool_name="transfer",
        tool_input={"amount_inr": 10_000, "beneficiary_id": "BOB-001"},
        state_before=None,
    )
    recorder.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS"},
        success=False,
    )
    recorder.record_outcome("INSUFFICIENT_FUNDS", success=False)
    return recorder.finish()


class TestScenarioF_MissingEvidence:
    def test_missing_state_report(self, pipeline):
        report = pipeline.analyze_execution(trace=_stateless_trace())
        _report_shape(report)

        assert "UNKNOWN (not recorded)" in " ".join(report.agent_action)
        assert any("could not be evaluated (missing evidence)" in lim
                   for lim in report.limitations)
        # missing state leaves requirements UNKNOWN — never assumed satisfied
        assert any(s == "UNKNOWN" for s in report.requirement_satisfaction.values())
        assert any("missing evidence" in u or "UNKNOWN" in u
                   for u in report.uncertainty)


# ----------------------------------------------------- G. conflicting evidence
class TestScenarioG_ConflictingEvidence:
    def test_conflicting_report(self, pipeline):
        env = BankEnvironment.from_dataset()
        recorder = ExecutionRecorder(
            task_id="TASK-SYN-G", task_text="Transfer ₹5,000 to beneficiary BOB-001.",
            agent_id="rule-based-agent", trace_id="TRC-SYN-G",
        )
        recorder.record_task_received()
        recorder.record_tool_invocation(
            action="transfer", tool_name="transfer",
            tool_input={"amount_inr": 5_000, "beneficiary_id": "BOB-001"},
            state_before=env.summary(),
        )
        first = recorder.record_tool_result(
            action="transfer", tool_name="transfer",
            tool_output={"success": True, "reason_code": "OK",
                         "account_id": "ACC-SYN-001"},
            success=True,
        )
        recorder.record_tool_result(
            action="transfer", tool_name="transfer",
            tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS",
                         "account_id": "ACC-SYN-001"},
            success=False, conflict_with_event_id=first.event_id,
        )
        recorder.record_outcome("CONFLICT", success=None)

        report = pipeline.analyze_execution(trace=recorder.finish())
        _report_shape(report)

        # the conflict is surfaced, never silently resolved
        conflicting = [a for a in report.requirements if a.conflicting_evidence]
        assert conflicting
        assert all(a.status is RequirementStatus.UNKNOWN for a in conflicting)


# -------------------------------------------------- H. unsupported explanation
class FakeAdversarialProvider:
    """Rationale 'model' that invents evidence — must be caught downstream."""

    name = "fake-adversarial"

    def generate(self, system_prompt, user_prompt):
        return json.dumps({
            "decision_rationale": "The transfer succeeded per evidence E99.",
            "deviation_explanation": "No problems; REQ-FAKE-001 was satisfied.",
            "counterfactual_summary": "Nothing to consider.",
            "evidence_summary": "All checks passed per E77.",
            "uncertainty_statement": "None at all.",
            "claims": [
                {"text": "Everything was fine and verified.",
                 "label": "FACT", "evidence_refs": ["E99"]},
            ],
        })


class TestScenarioH_UnsupportedExplanation:
    def test_adversarial_rationale_is_blocked(self):
        pipeline = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()),
            rationale_provider=FakeAdversarialProvider(),
        )
        _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        _report_shape(report)

        # Phase 6 quarantined the invented refs -> grounded=False
        assert report.rationale.grounded is False
        assert report.rationale.unsupported_claims
        # Phase 7 guardrail refuses to PASS an ungrounded rationale
        assert report.validation.guardrail_decision is GuardrailDecision.BLOCK
        assert any("not grounded upstream" in n for n in report.validation.notes)
        assert any("unverifiable" in lim for lim in report.limitations)


# ---------------------------------------------------- I. correct explanation
class FakeFaithfulProvider:
    """Rationale 'model' that restates the packet faithfully."""

    name = "fake-faithful"

    def __init__(self, ctx_lookup):
        self._lookup = ctx_lookup

    def generate(self, system_prompt, user_prompt):
        refs = self._lookup(user_prompt)
        return json.dumps({
            "decision_rationale": "The transfer was within the limit and succeeded.",
            "deviation_explanation": "No requirement violations were detected.",
            "counterfactual_summary": "No counterfactuals were generated.",
            "evidence_summary": "The recorded evidence covers the transfer.",
            "uncertainty_statement": "Assessments carry confidence below 1.0.",
            "claims": [
                {"text": "REQ-BEN-001 was satisfied.", "label": "EVIDENCE",
                 "evidence_refs": ["REQ-BEN-001"]},
                {"text": "REQ-BAL-001 was satisfied.", "label": "EVIDENCE",
                 "evidence_refs": ["REQ-BAL-001"]},
            ] + ([{"text": "REQ-AMT-001 was satisfied.", "label": "EVIDENCE",
                   "evidence_refs": ["REQ-AMT-001"]}] if refs else []),
        })


class TestScenarioI_CorrectExplanation:
    def test_faithful_rationale_passes_validation(self):
        pipeline = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()),
            rationale_provider=FakeFaithfulProvider(lambda prompt: "REQ-AMT-001" in prompt),
        )
        _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        _report_shape(report)

        assert report.rationale.llm_used is True
        assert report.rationale.grounded is True
        assert report.validation.overall_validation_status \
            is OverallValidationStatus.SUPPORTED
        assert report.validation.guardrail_decision is GuardrailDecision.PASS
        assert report.validation.contradicted_claim_count == 0
        assert len(report.supported_claims) == len(report.validation.verifications)


# ------------------------------------------------------------ determinism
class TestPipelineDeterminism:
    def test_two_fresh_pipelines_agree(self):
        # fresh agent+env per run -> identical inputs -> identical reports
        r1 = analyze_execution(
            "Transfer ₹75,000 to beneficiary BOB-001.",
            agent=RuleBasedAgent(BankEnvironment.from_dataset()))
        r2 = analyze_execution(
            "Transfer ₹75,000 to beneficiary BOB-001.",
            agent=RuleBasedAgent(BankEnvironment.from_dataset()))
        a, b = r1.model_dump(), r2.model_dump()
        for key in ("task_id", "trace_id", "timings_ms", "agent_execution_ms"):
            a.pop(key), b.pop(key)
        assert a == b
