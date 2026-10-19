"""Metric-to-trace linkage tests (Part 5).

The chain METRIC -> OBSERVED CHANGE -> RELEVANT TRACE EVENTS ->
EXPLANATION CLAIM -> EVIDENCE REFERENCES -> VALIDATION STATUS must be
complete, verbatim-faithful, and labelled association/hypothesis — never
causation, and never causal for CSAT/retention.
"""

import pytest

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.linkage import (
    LinkType,
    MetricChange,
    link_metric,
)


@pytest.fixture
def pipeline() -> TraceXAIPipeline:
    return TraceXAIPipeline(agent=RuleBasedAgent(BankEnvironment.from_dataset()))


@pytest.fixture
def violation_report(pipeline):
    _, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
    return report


class TestChainCompleteness:
    def test_scoped_chain_reaches_every_step(self, violation_report):
        link = link_metric(violation_report, MetricChange(
            metric="unsupported_claim_rate", before=0.4, after=0.0,
            unit="ratio", basis="measured",
            related_requirement_ids=["REQ-AMT-001"]))
        # [1] metric, [2] change, [3] events, [4] claims, refs, verdicts
        assert link.metric == "unsupported_claim_rate"
        assert link.metric_before == 0.4 and link.metric_after == 0.0
        assert link.observed_change.startswith("decreased")
        assert link.relevant_trace_events
        assert link.linked_claims
        for claim in link.linked_claims:
            assert claim.evidence_references
            assert claim.validation_status in ("SUPPORTED", "CONTRADICTED",
                                               "UNKNOWN")
            assert claim.verification_method
        text = link.to_text()
        for step in ("[1] METRIC", "[2] OBSERVED CHANGE",
                     "[3] RELEVANT TRACE EVENTS", "[4] EXPLANATION CLAIMS"):
            assert step in text

    def test_no_movement_is_described_honestly(self, violation_report):
        link = link_metric(violation_report, MetricChange(
            metric="contradiction_rate", before=0.0, after=0.0, unit="ratio",
            basis="measured", related_requirement_ids=["REQ-AMT-001"]))
        assert link.observed_change.startswith("no movement")

    def test_missing_before_after_is_not_invented(self, violation_report):
        link = link_metric(violation_report, MetricChange(
            metric="agent_execution_ms", after=0.3, unit="ms",
            basis="measured"))
        assert "no baseline" in link.observed_change
        assert "no movement is claimed" in link.observed_change


class TestVerbatimFidelity:
    def test_claims_and_verdicts_are_copied_not_recomputed(self,
                                                           violation_report):
        link = link_metric(violation_report, MetricChange(
            metric="unsupported_claim_rate", before=0.4, after=0.0,
            unit="ratio", basis="measured",
            related_requirement_ids=["REQ-AMT-001"]))
        verdicts = {v.claim: v.verification_status.value
                    for v in violation_report.validation.verifications}
        for claim in link.linked_claims:
            assert claim.claim in verdicts
            assert claim.validation_status == verdicts[claim.claim]

    def test_relevant_events_come_from_the_evidence_index(self,
                                                          violation_report):
        link = link_metric(violation_report, MetricChange(
            metric="x", related_requirement_ids=["REQ-AMT-001"]))
        for line in link.relevant_trace_events:
            assert line in violation_report.evidence


class TestScoping:
    def test_scoped_chain_is_tighter_than_unscoped(self, violation_report):
        scoped = link_metric(violation_report, MetricChange(
            metric="m", related_requirement_ids=["REQ-AMT-001"]))
        unscoped = link_metric(violation_report, MetricChange(metric="m"))
        assert len(scoped.relevant_trace_events) \
            < len(unscoped.relevant_trace_events)
        assert scoped.requirements_in_scope == ["REQ-AMT-001"]
        assert set(unscoped.requirements_in_scope) == {
            a.requirement_id for a in violation_report.requirements}
        assert any("broad" in l for l in unscoped.limitations)

    def test_unknown_requirement_is_reported_not_ignored(self,
                                                         violation_report):
        link = link_metric(violation_report, MetricChange(
            metric="m", related_requirement_ids=["REQ-NOPE-999"]))
        assert any("REQ-NOPE-999" in l for l in link.limitations)

    def test_linking_is_deterministic(self, violation_report):
        change = MetricChange(metric="m", before=1.0, after=2.0, unit="u",
                              related_requirement_ids=["REQ-AMT-001"])
        assert (link_metric(violation_report, change).model_dump()
                == link_metric(violation_report, change).model_dump())


class TestNoCausation:
    def test_links_are_never_labelled_causal(self, violation_report):
        assert "CAUSAL" not in {m.value for m in LinkType}
        for change in (MetricChange(metric="m", related_requirement_ids=[
                           "REQ-AMT-001"]),
                       MetricChange(metric="m")):
            link = link_metric(violation_report, change)
            assert link.link_type in (LinkType.ASSOCIATION, LinkType.HYPOTHESIS)
            assert any("not a causal" in n or "no causal" in n
                       for n in link.notes + [link.to_text()])

    def test_csat_movement_is_hypothesis_only(self, violation_report):
        for metric in ("csat_simulated", "CSAT", "retention_simulated"):
            link = link_metric(violation_report, MetricChange(
                metric=metric, before=3.0, after=4.5, unit="rating",
                basis="simulated"))
            assert link.link_type is LinkType.HYPOTHESIS
            assert link.linked_claims == []
            assert link.relevant_trace_events == []
            assert any("satisfaction proxy" in l for l in link.limitations)

    def test_csat_link_never_carries_claims_even_when_scoped(self,
                                                             violation_report):
        link = link_metric(violation_report, MetricChange(
            metric="csat", before=3.0, after=4.0, unit="rating",
            related_requirement_ids=["REQ-AMT-001"]))
        assert link.link_type is LinkType.HYPOTHESIS
        assert link.linked_claims == []


class TestLinkOnOtherScenarios:
    def test_link_on_unknown_requirement_report(self, pipeline):
        from trace_xai.core.recorder import ExecutionRecorder

        recorder = ExecutionRecorder(
            task_id="TASK-SYN-L", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
            agent_id="qwen3.5:4b", trace_id="TRC-SYN-L")
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

        link = link_metric(report, MetricChange(
            metric="unverifiable_rate", before=0.0, after=0.3, unit="ratio",
            basis="measured"))
        # the chain works even when the report is thin, and stays labelled
        assert link.link_type in (LinkType.ASSOCIATION, LinkType.HYPOTHESIS)
        for claim in link.linked_claims:
            assert claim.validation_status in ("SUPPORTED", "CONTRADICTED",
                                               "UNKNOWN")
