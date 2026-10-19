"""Metrics tests (Part 3): latency, cost, claim quality, CSAT.

Every assertion checks the honesty rules, not just the numbers:
- latency separates agent execution / XAI analysis / end-to-end, all in ms;
  unmeasured components are UNKNOWN (None), never silently 0;
- cost uses real provider-reported tokens; local models have no price so
  cost is UNKNOWN unless pricing is supplied (then ESTIMATED + source);
- unsupported-claim rate has an exact documented formula and decomposes into
  contradiction vs unverifiable; UNKNOWN is not a hallucination;
- CSAT/retention: real CSAT is NOT_COLLECTED (0 responses, no fabricated
  feedback); the simulated variant is deterministic, labelled and distinct
  from factual correctness.
"""

import pytest

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.metrics import (
    CSATRecorder,
    InjectedClaim,
    MetricBasis,
    ModelPricing,
    collect_metrics,
    evaluate_detection,
)
from trace_xai.validate.models import (
    ClaimVerification,
    VerificationMethod,
    VerificationStatus,
)


@pytest.fixture
def pipeline() -> TraceXAIPipeline:
    return TraceXAIPipeline(agent=RuleBasedAgent(BankEnvironment.from_dataset()))


@pytest.fixture
def violation_report(pipeline):
    _, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
    return report


# ---------------------------------------------------------------- latency
class TestLatencyMetrics:
    def test_run_separates_agent_from_xai(self, violation_report):
        lat = violation_report.metrics().latency
        assert lat.agent_execution_ms is not None
        assert lat.agent_execution_ms >= 0
        assert lat.agent_basis is MetricBasis.MEASURED
        assert lat.xai_analysis_ms is not None and lat.xai_analysis_ms > 0
        assert set(lat.xai_stage_ms) == {
            "decision", "deviation", "counterfactual", "rationale", "validation"}
        # end-to-end = agent + xai, in the same unit (ms)
        expected = round(lat.agent_execution_ms + lat.xai_analysis_ms, 3)
        assert lat.end_to_end_ms == pytest.approx(expected, abs=0.01)
        assert lat.end_to_end_basis is MetricBasis.MEASURED
        assert lat.unit == "ms"

    def test_direct_trace_analysis_is_unknown_not_zero(self, pipeline):
        _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        direct = pipeline.analyze_execution(trace=pipeline._traces[report.task_id])
        lat = direct.metrics().latency
        assert lat.agent_execution_ms is None          # UNKNOWN, not 0
        assert lat.agent_basis is MetricBasis.UNKNOWN
        assert lat.end_to_end_ms is None
        assert lat.end_to_end_basis is MetricBasis.UNKNOWN
        assert lat.xai_analysis_ms is not None         # XAI time still known
        assert any("UNKNOWN, not zero" in n for n in lat.notes)

    def test_xai_time_is_sum_of_stages(self, violation_report):
        lat = violation_report.metrics().latency
        assert lat.xai_analysis_ms == pytest.approx(
            sum(lat.xai_stage_ms.values()), abs=0.01)

    def test_metrics_are_deterministic_view(self, violation_report):
        a = violation_report.metrics().model_dump()
        b = violation_report.metrics().model_dump()
        assert a == b


# ------------------------------------------------------------------ cost
class TestCostMetrics:
    def test_no_model_used_zero_cost_is_measured(self, violation_report):
        cost = violation_report.metrics().cost
        assert cost.total_prompt_tokens == 0
        assert cost.total_completion_tokens == 0
        assert cost.estimated_cost_usd == 0.0
        assert cost.basis is MetricBasis.MEASURED
        assert cost.usage == []           # deterministic fallback: no calls
        assert any("deterministic fallback" in n for n in cost.notes)

    def test_unknown_pricing_is_reported_not_fabricated(self, violation_report):
        cost = violation_report.metrics().cost
        assert cost.pricing_known is False
        assert cost.estimated_cost_usd == 0.0 or cost.estimated_cost_usd is None
        assert cost.formula               # formula always documented

    def test_supplied_pricing_produces_labelled_estimate(self, violation_report):
        pricing = ModelPricing(
            usd_per_prompt_token=0.0001, usd_per_completion_token=0.0002,
            source="example rate card (caller-supplied, for demonstration)")
        cost = violation_report.metrics(pricing=pricing).cost
        assert cost.basis is MetricBasis.ESTIMATED
        assert cost.pricing_known is True
        assert cost.pricing_source == pricing.source
        expected = round(cost.total_prompt_tokens * 0.0001
                         + cost.total_completion_tokens * 0.0002, 8)
        assert cost.estimated_cost_usd == expected

    def test_usage_is_only_recorded_for_real_model_calls(self, violation_report):
        # default pipeline: rationale is the deterministic fallback -> no
        # provider-reported usage may appear
        assert violation_report.model_usage == []
        cost = violation_report.metrics().cost
        assert all(u.prompt_tokens is None or u.prompt_tokens >= 0
                   for u in cost.usage)


# --------------------------------------------------------- claim quality
class TestClaimQualityMetrics:
    def test_formulas_match_documented_definitions(self, violation_report):
        cq = violation_report.metrics().claim_quality
        v = violation_report.validation
        total = len(v.verifications)
        assert total > 0
        assert cq.unsupported_claim_rate == pytest.approx(
            (v.contradicted_claim_count + v.unknown_claim_count) / total, abs=1e-4)
        assert cq.contradiction_rate == pytest.approx(
            v.contradicted_claim_count / total, abs=1e-4)
        assert cq.unverifiable_rate == pytest.approx(
            v.unknown_claim_count / total, abs=1e-4)
        assert cq.formula and "unknown" in cq.formula

    def test_unknown_is_not_called_a_hallucination(self, violation_report):
        cq = violation_report.metrics().claim_quality
        assert any("not a hallucination" in n for n in cq.notes)
        # the two failure modes stay separate numbers
        assert (cq.contradicted_count == 0) != (cq.unknown_count == 0) or \
               cq.contradicted_count == cq.unknown_count

    def test_missing_evidence_scenario_reports_unverifiable(self, pipeline):
        recorder = ExecutionRecorder(
            task_id="TASK-SYN-M", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
            agent_id="qwen3.5:4b", trace_id="TRC-SYN-M")
        recorder.record_task_received()
        recorder.record_tool_invocation(
            action="transfer", tool_name="transfer",
            tool_input={"amount_inr": 10_000, "beneficiary_id": "BOB-001"},
            state_before=None)
        recorder.record_tool_result(
            action="transfer", tool_name="transfer",
            tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS"},
            success=False)
        recorder.record_outcome("INSUFFICIENT_FUNDS", success=False)
        report = pipeline.analyze_execution(trace=recorder.finish())
        cq = report.metrics().claim_quality
        # missing state leaves requirements UNKNOWN; the rationale's claims
        # about them are therefore not assertable as SUPPORTED-by-omission:
        # any UNKNOWN requirement must not be silently dropped from the
        # uncertainty record
        assert any(s == "UNKNOWN"
                   for s in report.requirement_satisfaction.values())
        assert cq.contradiction_rate == 0.0     # absence is not a contradiction

    def test_ungrounded_claims_are_counted_unverifiable(self, pipeline):
        # injected wrong refs -> the validator reports UNKNOWN verdicts; the
        # unverifiable rate must reflect them without calling them lies
        import json

        class FakeProvider:
            name = "fake-unknown"
            def generate(self, system_prompt, user_prompt):
                return json.dumps({
                    "decision_rationale": "The transfer succeeded per evidence E99.",
                    "deviation_explanation": "No problems; REQ-FAKE-001 ok.",
                    "counterfactual_summary": "Nothing to consider.",
                    "evidence_summary": "All checks pass per E77.",
                    "uncertainty_statement": "None at all.",
                    "claims": [{"text": "Everything was fine.",
                                "label": "FACT", "evidence_refs": ["E99"]}],
                })
        pipe = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()),
            rationale_provider=FakeProvider())
        _, report = pipe.run("Transfer ₹10,000 to beneficiary BOB-001.")
        cq = report.metrics().claim_quality
        assert cq.unknown_count > 0
        assert cq.unverifiable_rate > 0
        assert any("not a hallucination" in n for n in cq.notes)


# ------------------------------------------------------- detection metrics
class TestDetectionEvaluation:
    @staticmethod
    def _verdict(claim: str, status: VerificationStatus) -> ClaimVerification:
        return ClaimVerification(
            claim=claim, verification_status=status,
            verification_method=VerificationMethod.DETERMINISTIC_RULE,
            confidence=0.9, explanation="test verdict")

    def test_false_acceptance_and_detection_rates(self):
        verdicts = [
            self._verdict("wrong claim", VerificationStatus.CONTRADICTED),
            self._verdict("unverifiable claim", VerificationStatus.UNKNOWN),
            self._verdict("accepted wrong claim", VerificationStatus.SUPPORTED),
            self._verdict("true claim", VerificationStatus.SUPPORTED),
        ]
        truth = [
            InjectedClaim(text="wrong claim", ground_truth="CONTRADICTED",
                          origin="injected_wrong"),
            InjectedClaim(text="unverifiable claim", ground_truth="CONTRADICTED",
                          origin="injected_wrong"),
            InjectedClaim(text="accepted wrong claim", ground_truth="CONTRADICTED",
                          origin="injected_wrong"),
            InjectedClaim(text="true claim", ground_truth="SUPPORTED",
                          origin="synthetic_ground_truth"),
        ]
        m = evaluate_detection(verdicts, truth)
        assert m.wrong_claims == 3
        assert m.wrong_detected == 2             # CONTRADICTED + UNKNOWN
        assert m.wrong_accepted == 1             # the false acceptance
        assert m.false_acceptance_rate == pytest.approx(1 / 3, abs=1e-4)
        assert m.detection_rate == pytest.approx(2 / 3, abs=1e-4)
        assert m.correct_supported == 1
        assert m.missed_correct == 0
        assert any("no LLM judge" in n for n in m.notes)

    def test_no_wrong_claims_gives_none_not_zero(self):
        m = evaluate_detection(
            [self._verdict("ok", VerificationStatus.SUPPORTED)],
            [InjectedClaim(text="ok", ground_truth="SUPPORTED")])
        assert m.false_acceptance_rate is None   # not 0: nothing was tested
        assert m.detection_rate is None
        assert m.wrong_claims == 0

    def test_unmatched_ground_truth_is_disclosed(self):
        m = evaluate_detection(
            [self._verdict("seen", VerificationStatus.SUPPORTED)],
            [InjectedClaim(text="seen", ground_truth="SUPPORTED"),
             InjectedClaim(text="never verified", ground_truth="CONTRADICTED")])
        assert any("did not appear" in n for n in m.notes)


# ------------------------------------------------------------------ CSAT
class TestCSAT:
    def test_real_csat_is_not_collected(self, violation_report):
        csat = violation_report.metrics().csat
        assert csat.status == "NOT_COLLECTED"
        assert csat.responses == 0
        assert csat.mean_rating is None          # no fabricated average
        assert csat.simulated is None            # simulation is opt-in
        assert csat.scale_min == 1 and csat.scale_max == 5
        assert csat.scale_definition

    def test_simulated_csat_is_labelled_and_deterministic(self, violation_report):
        a = violation_report.metrics(simulate_csat=True).csat
        b = violation_report.metrics(simulate_csat=True).csat
        assert a.simulated.label.startswith("SIMULATED")
        assert a.simulated_retention.label.startswith("SIMULATED")
        assert a.simulated.simulated_mean_rating == b.simulated.simulated_mean_rating
        assert 1.0 <= a.simulated.simulated_mean_rating <= 5.0
        assert 0.0 <= a.simulated_retention.estimated_retention_rate <= 1.0
        assert a.simulated.formula and a.simulated_retention.formula
        assert a.responses == 0                 # real count stays 0
        assert a.mean_rating is None            # real mean stays unfilled

    def test_blocked_guardrail_lowers_simulated_csat(self, pipeline):
        # a clean run vs an adversarial (BLOCK) run must differ
        _, clean = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        import json
        class FakeProvider:
            name = "fake"
            def generate(self, system_prompt, user_prompt):
                return json.dumps({
                    "decision_rationale": "The transfer succeeded per evidence E99.",
                    "deviation_explanation": "No problems; REQ-FAKE-001 ok.",
                    "counterfactual_summary": "Nothing to consider.",
                    "evidence_summary": "All checks pass per E77.",
                    "uncertainty_statement": "None at all.",
                    "claims": [{"text": "Everything was fine.",
                                "label": "FACT", "evidence_refs": ["E99"]}],
                })
        blocked_pipe = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()),
            rationale_provider=FakeProvider())
        _, blocked = blocked_pipe.run("Transfer ₹10,000 to beneficiary BOB-001.")
        clean_csat = clean.metrics(simulate_csat=True).csat
        blocked_csat = blocked.metrics(simulate_csat=True).csat
        assert blocked_csat.simulated.simulated_mean_rating \
            < clean_csat.simulated.simulated_mean_rating

    def test_csat_is_distinct_from_factual_correctness(self, violation_report):
        m = violation_report.metrics(simulate_csat=True)
        # claim quality (factual) and csat (satisfaction proxy) are separate
        # sections with separate definitions
        assert m.claim_quality.formula != m.csat.simulated.formula
        assert m.csat.simulated.inputs["guardrail"]

    def test_recorder_only_reports_what_was_recorded(self, violation_report):
        rec = CSATRecorder()
        empty = rec.metrics()
        assert empty.responses == 0 and empty.status == "NOT_COLLECTED"
        rec.record(4)
        rec.record(5)
        filled = rec.metrics(report=violation_report, simulate=True)
        assert filled.responses == 2
        assert filled.mean_rating == pytest.approx(4.5)
        assert filled.status == "COLLECTED"
        assert filled.simulated is not None      # labelled simulation attached

    def test_recorder_rejects_out_of_scale(self):
        rec = CSATRecorder()
        with pytest.raises(ValueError):
            rec.record(6)
        with pytest.raises(ValueError):
            rec.record(0)


# -------------------------------------------------------------- integration
class TestMetricsIntegration:
    def test_collect_metrics_from_any_report(self, violation_report):
        bundle = collect_metrics(violation_report)
        assert bundle.latency.unit == "ms"
        assert bundle.cost.unit == "USD"
        assert bundle.claim_quality.total_claims >= 1
        assert bundle.csat.scale_min == 1

    def test_metrics_do_not_mutate_the_report(self, violation_report):
        before = violation_report.model_dump()
        violation_report.metrics(simulate_csat=True)
        after = violation_report.model_dump()
        assert before == after
