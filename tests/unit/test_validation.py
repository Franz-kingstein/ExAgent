"""Phase 7 tests: explanation validation (Q4).

Required cases:
1. Correct explanation          4. Contradicted explanation
2. Partially correct explanation 5. Missing evidence
3. Unsupported explanation      6. Conflicting evidence

Plus: deliberately incorrect explanations must be caught, the LLM judge must
never overrule deterministic verdicts, and a correct honest explanation must
NOT be contradicted (false-positive guard).
"""

import json

import pytest

from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.rationale import RationaleGenerator, build_context
from trace_xai.rationale.models import ClaimLabel, RationaleClaim, StructuredRationale
from trace_xai.validate import (
    ExplanationValidator,
    GuardrailDecision,
    LLMJudge,
    OverallValidationStatus,
    ValidationReport,
    VerificationMethod,
    VerificationStatus,
    validate_explanation,
)

_SUP = VerificationStatus.SUPPORTED
_CON = VerificationStatus.CONTRADICTED
_UNK = VerificationStatus.UNKNOWN


# --------------------------------------------------------------- helpers
def make_rationale(claims: list[dict], grounded: bool = True) -> StructuredRationale:
    """Minimal rationale; sections carry no citations unless overridden."""
    return StructuredRationale(
        task_id="TASK-SYN-0001", trace_id="TRC-SYN-0001",
        agent_id="rule-based-agent", action="TRANSFER",
        decision_rationale="Deterministic section text.",
        deviation_explanation="Deterministic section text.",
        counterfactual_summary="Deterministic section text.",
        evidence_summary="Deterministic section text.",
        uncertainty_statement="Deterministic section text.",
        claims=[RationaleClaim(**c) for c in claims],
        grounded=grounded,
        provider="test",
    )


def ref_for(ctx, substring: str) -> str:
    for entry in ctx.evidence:
        if substring in entry.field:
            return entry.ref
    raise AssertionError(f"no evidence entry matches {substring!r}")


class FakeJudge:
    name = "fake-judge"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


@pytest.fixture(scope="module")
def trace_ctx():
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
    return trace, build_context(trace)


# ------------------------------------------------- 1. correct explanation
class TestCase1CorrectExplanation:
    def test_all_claims_supported_and_pass(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "The transfer amount of INR 75,000 exceeded the per-transaction "
                     "limit.", "label": "EVIDENCE",
             "evidence_refs": [ref_for(ctx, "amount_inr"), ref_for(ctx, "per_txn_limits")]},
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
            {"text": "The beneficiary was verified.", "label": "EVIDENCE",
             "evidence_refs": [ref_for(ctx, "beneficiaries_verified")]},
        ])
        report = validate_explanation(rationale, ctx)

        assert isinstance(report, ValidationReport)
        assert report.supported_claim_count == 3
        assert report.contradicted_claim_count == 0
        assert report.unknown_claim_count == 0
        assert report.unsupported_claim_rate == 0.0
        assert report.overall_validation_status is OverallValidationStatus.SUPPORTED
        assert report.guardrail_decision is GuardrailDecision.PASS
        assert all(v.verification_method is VerificationMethod.DETERMINISTIC_RULE
                   for v in report.verifications)

    def test_every_verdict_carries_evidence_and_explanation(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
        ])
        report = validate_explanation(rationale, ctx)
        verdict = report.verifications[0]
        assert verdict.evidence, "verdict must show the evidence it used"
        assert verdict.explanation
        assert 0.0 <= verdict.confidence <= 1.0


# --------------------------------------- 2. partially correct explanation
class TestCase2PartiallyCorrect:
    def test_mixed_verdicts_rate_and_flag(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "The transfer amount of INR 75,000 exceeded the per-transaction "
                     "limit.", "label": "EVIDENCE",
             "evidence_refs": [ref_for(ctx, "amount_inr"), ref_for(ctx, "per_txn_limits")]},
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
            {"text": "The agent carefully considered alternative recipients before "
                     "acting.", "label": "INFERENCE",
             "evidence_refs": [ref_for(ctx, "amount_inr")]},
        ])
        report = validate_explanation(rationale, ctx)

        assert report.supported_claim_count == 2
        assert report.unknown_claim_count == 1          # unverifiable inference
        assert report.contradicted_claim_count == 0
        assert report.unsupported_claim_rate == pytest.approx(1 / 3, abs=0.01)
        assert report.overall_validation_status is OverallValidationStatus.INSUFFICIENT_EVIDENCE
        assert report.guardrail_decision is GuardrailDecision.FLAG


# ------------------------------------------ 3. unsupported explanation
class TestCase3Unsupported:
    def test_invented_and_missing_citations_never_supported(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "The balance before the transfer was 999,999.",
             "label": "FACT", "evidence_refs": ["E99"]},          # invented ref
            {"text": "The transfer was processed instantly.",
             "label": "FACT", "evidence_refs": []},               # no citation
            {"text": "REQ-LIM-777 was satisfied.", "label": "FACT",
             "evidence_refs": ["REQ-LIM-777"]},                   # unknown requirement
        ])
        report = validate_explanation(rationale, ctx)

        assert report.supported_claim_count == 0
        assert report.unknown_claim_count == 3
        assert report.unsupported_claim_rate == 1.0
        assert report.overall_validation_status is OverallValidationStatus.INSUFFICIENT_EVIDENCE
        assert report.guardrail_decision is GuardrailDecision.BLOCK
        for verdict in report.verifications:
            assert verdict.verification_status is not _SUP
            assert verdict.verification_method is VerificationMethod.GUARDRAIL


# ----------------------------------------- 4. contradicted explanation
class TestCase4Contradicted:
    def test_within_limit_claim_contradicted_by_values(self, trace_ctx):
        # the spec's second example, verbatim
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "The transfer was within the ₹50,000 limit.",
             "label": "FACT",
             "evidence_refs": [ref_for(ctx, "amount_inr"), ref_for(ctx, "per_txn_limits")]},
        ])
        report = validate_explanation(rationale, ctx)

        verdict = report.verifications[0]
        assert verdict.verification_status is _CON
        assert verdict.verification_method is VerificationMethod.DETERMINISTIC_RULE
        assert "75000" in verdict.explanation and "50000" in verdict.explanation
        assert report.contradicted_claim_count == 1
        assert report.overall_validation_status is OverallValidationStatus.CONTRADICTED
        assert report.guardrail_decision is GuardrailDecision.BLOCK

    def test_verified_claim_contradicted_when_evidence_says_false(self, trace_ctx):
        import dataclasses

        _, ctx = trace_ctx
        # simulate an unverified beneficiary without mutating the shared fixture
        flipped = [
            dataclasses.replace(e, value=False)
            if "beneficiaries_verified" in e.field else e
            for e in ctx.evidence
        ]
        ctx_copy = dataclasses.replace(ctx, evidence=flipped)
        rationale = make_rationale([
            {"text": "The beneficiary was verified.", "label": "FACT",
             "evidence_refs": [ref_for(ctx_copy, "beneficiaries_verified")]},
        ])
        report = validate_explanation(rationale, ctx_copy)
        assert report.verifications[0].verification_status is _CON
        assert report.guardrail_decision is GuardrailDecision.BLOCK


# --------------------------------------------------- 5. missing evidence
def _stateless_trace():
    recorder = ExecutionRecorder(
        task_id="task-stateless", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b",
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


class TestCase5MissingEvidence:
    def test_verification_claim_with_unknown_evidence_is_unknown(self):
        # the spec's first example, verbatim
        ctx = build_context(_stateless_trace())
        any_ref = ctx.evidence[0].ref
        rationale = make_rationale([
            {"text": "The agent transferred ₹75,000 because the beneficiary was "
                     "verified.", "label": "EVIDENCE", "evidence_refs": [any_ref]},
        ])
        report = validate_explanation(rationale, ctx)

        verdict = report.verifications[0]
        assert verdict.verification_status is _UNK          # NOT supported
        assert "no (usable) verification value" in verdict.explanation
        assert report.guardrail_decision is not GuardrailDecision.PASS

    def test_unresolvable_citation_is_unknown_not_supported(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "The environment confirmed the beneficiary status.",
             "label": "FACT", "evidence_refs": ["E77"]},
        ])
        report = validate_explanation(rationale, ctx)
        assert report.verifications[0].verification_status is _UNK
        assert "do not resolve" in report.verifications[0].explanation


# ---------------------------------------------- 6. conflicting evidence
class TestCase6ConflictingEvidence:
    def test_conflicting_success_values_yield_unknown_not_verdict(self):
        from trace_xai.rationale.context import EvidenceEntry, RationaleContext

        ctx = RationaleContext(
            task_id="T", trace_id="TR", task_text="t", agent_id="a", synthetic=True,
        )
        ctx.evidence = [
            EvidenceEntry(ref="E1", event_id="TR-E002", field="tool_output.success",
                          value=True, source="decision"),
            EvidenceEntry(ref="E2", event_id="TR-E003", field="tool_output.success",
                          value=False, source="decision"),
        ]
        rationale = make_rationale([
            {"text": "The transfer succeeded.", "label": "FACT",
             "evidence_refs": ["E1", "E2"]},
        ])
        report = validate_explanation(rationale, ctx)

        verdict = report.verifications[0]
        assert verdict.verification_status is _UNK
        assert "conflicting" in verdict.explanation
        assert report.overall_validation_status is OverallValidationStatus.INSUFFICIENT_EVIDENCE


# ------------------------------------------------------- judge behavior
class TestJudgeBehavior:
    def _semantic_case(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "The payee was approved for transfers.",
             "label": "INFERENCE",
             "evidence_refs": [ref_for(ctx, "beneficiaries_verified")]},
        ])
        return rationale, ctx

    def test_judge_resolves_semantic_claims(self, trace_ctx):
        rationale, ctx = self._semantic_case(trace_ctx)
        judge = FakeJudge([json.dumps({
            "status": "SUPPORTED", "confidence": 0.9,
            "explanation": "evidence records the beneficiary as verified",
        })])
        report = ExplanationValidator(judge_provider=judge).validate(rationale, ctx)

        verdict = report.verifications[0]
        assert verdict.verification_status is _SUP
        assert verdict.verification_method is VerificationMethod.LLM_JUDGE
        assert verdict.confidence <= 0.75          # capped: never ground truth
        assert report.judge_used is True
        # judge saw only claim + cited evidence
        _, user = judge.calls[0]
        assert "CITED EVIDENCE" in user and "beneficiaries_verified" in user

    def test_deterministic_verdict_wins_judge_not_consulted(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "The transfer amount of INR 75,000 exceeded the per-transaction "
                     "limit.", "label": "FACT",
             "evidence_refs": [ref_for(ctx, "amount_inr"), ref_for(ctx, "per_txn_limits")]},
        ])
        judge = FakeJudge([json.dumps({"status": "CONTRADICTED", "confidence": 1.0,
                                       "explanation": "wrong on purpose"})])
        report = ExplanationValidator(judge_provider=judge).validate(rationale, ctx)

        assert judge.calls == []                   # never consulted
        assert report.verifications[0].verification_status is _SUP
        assert report.verifications[0].verification_method is VerificationMethod.DETERMINISTIC_RULE

    def test_judge_support_without_evidence_is_forced_unknown(self):
        judge = LLMJudge(FakeJudge([json.dumps({"status": "SUPPORTED",
                                                "confidence": 1.0,
                                                "explanation": "vibes"})]))
        verdict = judge.judge("Anything at all.", [])
        assert verdict is not None
        status, _confidence, explanation = verdict
        assert status is _UNK
        assert "no cited evidence" in explanation

    def test_unusable_judge_output_leaves_claim_unknown(self, trace_ctx):
        rationale, ctx = self._semantic_case(trace_ctx)
        judge = FakeJudge(["not json"])
        report = ExplanationValidator(judge_provider=judge).validate(rationale, ctx)
        assert report.verifications[0].verification_status is _UNK
        assert report.judge_used is False


# ------------------------------------------------ extraction & skipping
class TestExtraction:
    def test_citable_section_sentences_are_verified(self, trace_ctx):
        _, ctx = trace_ctx
        # model-produced prose: the hallucination net scans sections even when
        # the claims array is empty, so a buried citable assertion is caught
        rationale = make_rationale([])
        rationale.llm_used = True
        rationale.evidence_summary = (
            "The amount of INR 75,000 exceeded the per-transaction limit "
            f"({ref_for(ctx, 'amount_inr')}, {ref_for(ctx, 'per_txn_limits')})."
        )
        report = validate_explanation(rationale, ctx)
        assert len(report.verifications) == 1
        assert report.verifications[0].verification_status is _SUP

    def test_unknown_labelled_statements_are_skipped(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "Verification status is UNKNOWN for this trace.",
             "label": "UNKNOWN", "evidence_refs": []},
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
        ])
        report = validate_explanation(rationale, ctx)
        assert len(report.verifications) == 1
        assert any("skipped" in n for n in report.notes)


# ------------------------------------------------------- guardrail edge
class TestGuardrail:
    def test_ungrounded_input_can_never_pass(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
        ], grounded=False)
        report = validate_explanation(rationale, ctx)
        assert report.guardrail_decision is GuardrailDecision.BLOCK
        assert any("not grounded upstream" in n for n in report.notes)

    def test_no_claims_flags(self, trace_ctx):
        _, ctx = trace_ctx
        report = validate_explanation(make_rationale([]), ctx)
        assert report.guardrail_decision is GuardrailDecision.FLAG
        assert any("no factual claims" in n for n in report.notes)

    def test_rate_and_counts_are_consistent(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
            {"text": "REQ-BEN-001 was satisfied.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-BEN-001"]},
            {"text": "REQ-BAL-001 was violated.", "label": "FACT",
             "evidence_refs": ["REQ-BAL-001"]},                    # contradicted
            {"text": "The transfer completed instantly.", "label": "FACT",
             "evidence_refs": [ref_for(ctx, "amount_inr")]},       # unknown
        ])
        report = validate_explanation(rationale, ctx)
        assert (report.supported_claim_count, report.contradicted_claim_count,
                report.unknown_claim_count) == (2, 1, 1)
        assert report.unsupported_claim_rate == pytest.approx(0.5)
        assert report.guardrail_decision is GuardrailDecision.BLOCK


# -------------------------------------------------------------- end-to-end
class TestEndToEnd:
    def test_honest_fallback_rationale_is_not_contradicted(self):
        # false-positive guard: a correct Phase 6 rationale must not be judged wrong
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        ctx = build_context(trace)
        rationale = RationaleGenerator().generate(trace, context=ctx)
        report = validate_explanation(rationale, ctx)

        assert report.contradicted_claim_count == 0
        assert report.overall_validation_status is not OverallValidationStatus.CONTRADICTED

    def test_determinism(self, trace_ctx):
        _, ctx = trace_ctx
        rationale = make_rationale([
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
        ])
        first = validate_explanation(rationale, ctx)
        second = validate_explanation(rationale, ctx)
        assert first.model_dump() == second.model_dump()
