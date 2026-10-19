"""Optional integration tests: LLM-as-judge against local Ollama.

Skipped automatically when the daemon is unreachable. Assertions are
structural (statuses, caps, method labels) — never on exact judgment
phrasing, because a model verdict is inherently non-deterministic in content.
"""

import urllib.request

import pytest

from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.rationale import OllamaRationaleProvider, build_context
from trace_xai.rationale.models import ClaimLabel, RationaleClaim, StructuredRationale
from trace_xai.validate import (
    ExplanationValidator,
    VerificationMethod,
    VerificationStatus,
)


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="Local Ollama not reachable")


@pytest.fixture(scope="module")
def ctx():
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
    return build_context(trace)


def _ref(ctx, substring: str) -> str:
    return next(e.ref for e in ctx.evidence if substring in e.field)


def _rationale(claims):
    return StructuredRationale(
        task_id="TASK-SYN-0001", trace_id="TRC-SYN-0001",
        agent_id="rule-based-agent", action="TRANSFER",
        decision_rationale="Section.", deviation_explanation="Section.",
        counterfactual_summary="Section.", evidence_summary="Section.",
        uncertainty_statement="Section.",
        claims=[RationaleClaim(**c) for c in claims],
        provider="test",
    )


def test_judge_semantic_paraphrase_respects_evidence(ctx):
    """Paraphrase the deterministic rules cannot decide goes to the judge."""
    beneficiary_ref = _ref(ctx, "beneficiaries_verified")
    rationale = _rationale([
        {"text": "The payee had been approved for receiving transfers.",
         "label": "INFERENCE",
         "evidence_refs": [beneficiary_ref]},
    ])
    validator = ExplanationValidator(
        judge_provider=OllamaRationaleProvider(model="qwen3.5:4b"))
    report = validator.validate(rationale, ctx)

    verdict = report.verifications[0]
    # the judge must engage (or stay UNKNOWN) — and cap its own authority
    assert verdict.verification_method is VerificationMethod.LLM_JUDGE
    assert verdict.confidence <= 0.75
    assert verdict.verification_status in set(VerificationStatus)
    # the cited evidence really does record verification=True, so SUPPORTED is
    # the correct answer; accept UNKNOWN (conservative) but never CONTRADICTED
    assert verdict.verification_status is not VerificationStatus.CONTRADICTED


def test_judge_cannot_rescue_an_unsupported_number_claim(ctx):
    """A false number claim must stay busted even with a judge available."""
    rationale = _rationale([
        {"text": "The balance before the transfer was 999,999.",
         "label": "FACT",
         "evidence_refs": [_ref(ctx, "balances"), _ref(ctx, "per_txn_limits")]},
    ])
    validator = ExplanationValidator(
        judge_provider=OllamaRationaleProvider(model="qwen3.5:4b"))
    report = validator.validate(rationale, ctx)

    verdict = report.verifications[0]
    # deterministic rule 6 catches it before any judge sees it
    assert verdict.verification_status is VerificationStatus.CONTRADICTED
    assert verdict.verification_method is VerificationMethod.DETERMINISTIC_RULE
    assert report.guardrail_decision.value == "BLOCK"


def test_deterministic_path_is_preferred_for_decidable_claims(ctx):
    """With a judge configured, decidable claims still use rules (no API cost)."""
    rationale = _rationale([
        {"text": "The transfer amount of INR 75,000 exceeded the per-transaction "
                 "limit.", "label": "FACT",
         "evidence_refs": [_ref(ctx, "amount_inr"), _ref(ctx, "per_txn_limits")]},
    ])
    validator = ExplanationValidator(
        judge_provider=OllamaRationaleProvider(model="qwen3.5:4b"))
    report = validator.validate(rationale, ctx)
    assert report.verifications[0].verification_method is VerificationMethod.DETERMINISTIC_RULE
    assert report.judge_used is False
