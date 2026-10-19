"""Explanation validation pipeline (Q4).

    generated explanation
            |
     claim extraction        (extract.py: claims array + citable sentences)
            |
     evidence retrieval      (RationaleContext: E-numbers, REQ/ANOM/CF lines)
            |
     claim verification      deterministic rules -> lexical NLI -> LLM judge
            |
     SUPPORTED / CONTRADICTED / UNKNOWN
            |
     guardrail decision      PASS / FLAG / BLOCK

Design principles:

- Deterministic verification first and last: the LLM judge only touches
  claims that every deterministic method left UNKNOWN, and its verdict is
  capped in confidence and guardrailed (no evidence -> no support).
- UNKNOWN-labelled claims from Phase 6 are uncertainty *statements*, not
  assertions; they are skipped (recorded), never counted as supported.
- The validator judges; it never rewrites explanations.
- `grounded=False` input (Phase 6 quarantined citations upstream) can never
  PASS the guardrail.
"""

from __future__ import annotations

import re

from ..rationale.context import RationaleContext
from ..rationale.models import StructuredRationale
from .extract import ExtractedClaim, extract_claims
from .judge import LLMJudge
from .models import (
    ClaimVerification,
    GuardrailDecision,
    OverallValidationStatus,
    ValidationReport,
    VerificationMethod,
    VerificationStatus,
)
from .rules import content_overlap, extract_operands, lexical_entailment, run_rules

_SUP = VerificationStatus.SUPPORTED
_CON = VerificationStatus.CONTRADICTED
_UNK = VerificationStatus.UNKNOWN

_REQ_STATUS_LINE = re.compile(
    r"^(REQ-[A-Z]+-\d+):\s+(SATISFIED|VIOLATED|UNKNOWN)\b", re.IGNORECASE)


class ExplanationValidator:
    """Verifies explanation claims against the recorded evidence."""

    name = "explanation-validator-v1"

    def __init__(self, judge_provider=None, block_rate_threshold: float = 0.5,
                 flag_rate_threshold: float = 0.0) -> None:
        self._judge = LLMJudge(judge_provider) if judge_provider is not None else None
        self._block_rate = block_rate_threshold
        self._flag_rate = flag_rate_threshold

    # ------------------------------------------------------------- public
    def validate(self, rationale: StructuredRationale,
                 context: RationaleContext) -> ValidationReport:
        notes: list[str] = []
        claims = extract_claims(rationale)
        assertions = [c for c in claims if (c.label or "").upper() != "UNKNOWN"]
        skipped = len(claims) - len(assertions)
        if skipped:
            notes.append(f"{skipped} uncertainty statement(s) skipped "
                         f"(not factual assertions)")

        req_statuses = _requirement_statuses(context)
        verifications: list[ClaimVerification] = []
        judge_used = False

        for claim in assertions:
            verification, judged = self._verify(claim, context, req_statuses)
            verifications.append(verification)
            judge_used = judge_used or judged

        supported = sum(1 for v in verifications if v.verification_status is _SUP)
        contradicted = sum(1 for v in verifications if v.verification_status is _CON)
        unknown = sum(1 for v in verifications if v.verification_status is _UNK)
        total = len(verifications)
        rate = (contradicted + unknown) / total if total else 0.0

        if contradicted:
            overall = OverallValidationStatus.CONTRADICTED
        elif unknown or not total:
            overall = OverallValidationStatus.INSUFFICIENT_EVIDENCE
        else:
            overall = OverallValidationStatus.SUPPORTED

        guardrail = self._guardrail(contradicted, unknown, total, rate, rationale, notes)

        return ValidationReport(
            task_id=rationale.task_id, trace_id=rationale.trace_id,
            verifications=verifications,
            supported_claim_count=supported,
            contradicted_claim_count=contradicted,
            unknown_claim_count=unknown,
            unsupported_claim_rate=round(rate, 3),
            overall_validation_status=overall,
            guardrail_decision=guardrail,
            judge_used=judge_used,
            notes=notes,
        )

    # ------------------------------------------------------------ internals
    def _verify(self, claim: ExtractedClaim, context: RationaleContext,
                req_statuses: dict[str, str]) -> tuple[ClaimVerification, bool]:
        allowed = context.allowed_refs()
        unresolvable = [r for r in claim.refs if r not in allowed]
        if not claim.refs:
            return ClaimVerification(
                claim=claim.text, evidence=[],
                verification_status=_UNK,
                verification_method=VerificationMethod.GUARDRAIL,
                confidence=0.1,
                explanation="claim cites no evidence; nothing can verify it "
                            "-> UNKNOWN (never SUPPORTED)",
            ), False
        if unresolvable:
            return ClaimVerification(
                claim=claim.text, evidence=[],
                verification_status=_UNK,
                verification_method=VerificationMethod.GUARDRAIL,
                confidence=0.1,
                explanation=f"citation(s) {unresolvable} do not resolve to the "
                            f"recorded evidence -> UNKNOWN",
            ), False

        evidence_lines, evidence_values = _retrieve(claim.refs, context)

        # 1-2. deterministic rules (preferred)
        ops = extract_operands(evidence_values)
        hit = run_rules(claim.text, ops, req_statuses)
        if hit is not None:
            return ClaimVerification(
                claim=claim.text, evidence=evidence_lines,
                verification_status=hit.status,
                verification_method=VerificationMethod.DETERMINISTIC_RULE,
                confidence=hit.confidence,
                explanation=hit.explanation,
            ), False

        # 3. lexical entailment backstop (can only contradict, never support)
        nli = lexical_entailment(claim.text, " | ".join(evidence_lines))
        if nli is not None:
            return ClaimVerification(
                claim=claim.text, evidence=evidence_lines,
                verification_status=nli.status,
                verification_method=VerificationMethod.LEXICAL_NLI,
                confidence=nli.confidence,
                explanation=nli.explanation,
            ), False

        # 4. LLM judge (semantic claims only; optional; never overrules rules)
        if self._judge is not None:
            verdict = self._judge.judge(claim.text, evidence_lines)
            if verdict is not None:
                status, confidence, explanation = verdict
                return ClaimVerification(
                    claim=claim.text, evidence=evidence_lines,
                    verification_status=status,
                    verification_method=VerificationMethod.LLM_JUDGE,
                    confidence=confidence,
                    explanation=explanation,
                ), True

        # 5. nothing decided the claim -> UNKNOWN (explicitly not supported)
        overlap = content_overlap(claim.text, " | ".join(evidence_lines))
        return ClaimVerification(
            claim=claim.text, evidence=evidence_lines,
            verification_status=_UNK,
            verification_method=VerificationMethod.LEXICAL_NLI,
            confidence=0.3,
            explanation=f"no deterministic rule matched and lexical overlap is "
                        f"{overlap:.2f}; evidence does not settle the claim "
                        f"-> UNKNOWN",
        ), False

    def _guardrail(self, contradicted: int, unknown: int, total: int,
                   rate: float, rationale: StructuredRationale,
                   notes: list[str]) -> GuardrailDecision:
        """Strict policy: false facts block; uncertainty flags; clean passes."""
        if not rationale.grounded:
            notes.append("input rationale was not grounded upstream (Phase 6 "
                         "quarantined citations); cannot PASS")
            return GuardrailDecision.BLOCK
        if contradicted:
            return GuardrailDecision.BLOCK
        if total and rate >= self._block_rate:
            notes.append(f"unsupported claim rate {rate:.2f} >= "
                         f"{self._block_rate}; too little of the explanation "
                         f"is evidenced")
            return GuardrailDecision.BLOCK
        if unknown and rate > self._flag_rate:
            return GuardrailDecision.FLAG
        if not total:
            notes.append("no factual claims to validate")
            return GuardrailDecision.FLAG
        return GuardrailDecision.PASS


# ---------------------------------------------------------- retrieval
def _retrieve(refs: list[str], context: RationaleContext) -> tuple[
        list[str], list[tuple[str, object]]]:
    """Cited refs -> rendered evidence lines + (field, value) operand pairs."""
    lines: list[str] = []
    values: list[tuple[str, object]] = []
    by_ref = {e.ref: e for e in context.evidence}
    for ref in refs:
        entry = by_ref.get(ref)
        if entry is not None:
            lines.append(f"{ref} = {entry.event_id}.{entry.field} = {entry.value!r}")
            values.append((entry.field, entry.value))
            continue
        # domain ids: REQ-/ANOM-/CF- -> the matching context line
        for line in context.requirement_matching + context.trajectory_anomalies \
                + context.counterfactuals:
            if ref in line:
                lines.append(f"{ref} = {line}")
                break
    return lines, values


def _requirement_statuses(context: RationaleContext) -> dict[str, str]:
    """REQ-id -> recorded status from the Q1 matching lines."""
    statuses: dict[str, str] = {}
    for line in context.requirement_matching:
        match = _REQ_STATUS_LINE.match(line.strip())
        if match:
            statuses[match.group(1)] = match.group(2).upper()
    return statuses


def validate_explanation(rationale: StructuredRationale,
                         context: RationaleContext, judge_provider=None) -> \
        ValidationReport:
    """One-call convenience API."""
    return ExplanationValidator(judge_provider=judge_provider).validate(
        rationale, context)
