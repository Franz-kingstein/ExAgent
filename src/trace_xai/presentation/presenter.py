"""Build an audience-adapted explanation from a validated ``FinalXAIReport``.

Invariants enforced here (and asserted by tests/unit/test_presentation.py):

1. Verification statuses, claim texts, counts, guardrail and unsupported-
   claim rate are **copied** from the validated report, never recomputed.
2. Every mode carries the full evidence index and per-claim evidence lines,
   so evidence references stay accessible in all four modes.
3. Every mode carries the recorded rationale/deviation text verbatim in
   ``recorded``, so simplifying wording never drops source facts.
4. Each presented item is tagged FACT / INFERENCE / UNKNOWN, so observed
   facts and inferences stay distinguishable in every mode.
5. No model is called while re-wording (deterministic pure function).

The wording layer adapts *presentation* only — it is not an explainability
algorithm and does not change any conclusion.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..explain.models import EvidenceItem
from .modes import (
    MODE_AUDIENCE,
    MODE_DESCRIPTION,
    ClaimKind,
    INTRO,
    PresentationMode,
    status_words,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..pipeline import FinalXAIReport

__all__ = [
    "PresentationMode",
    "ClaimKind",
    "PresentedLine",
    "PresentedBlock",
    "PresentedRequirement",
    "PresentedClaim",
    "PresentedExplanation",
    "present_report",
    "present_all",
]


# Recorded reason codes -> plain language (deterministic translation table;
# unknown codes are passed through unchanged rather than guessed).
_REASON_PLAIN = {
    "OK": "it went through",
    "LIMIT_EXCEEDED": "it was over the allowed money limit",
    "INSUFFICIENT_FUNDS": "the account did not have enough money",
    "BENEFICIARY_UNVERIFIED": "the receiver was not verified",
    "BENEFICIARY_NOT_LINKED": "the receiver was not linked to this account",
    "BENEFICIARY_NOT_FOUND": "the receiver could not be found",
    "INVALID_AMOUNT": "the amount was not valid",
    "ACCOUNT_NOT_FOUND": "the account could not be found",
    "ACCOUNT_NOT_ACTIVE": "the account is not active",
    "UNPARSEABLE_TASK": "the request could not be understood",
    "TASK_NOT_SUPPORTED": "that kind of request is not supported",
    "INTERNAL_ERROR": "an internal error happened",
}

_ACTION_PAIR = re.compile(r"^([a-z_]+):\s*(.*)$")


class PresentedLine(BaseModel):
    """One rendered line with its epistemic class and any cited refs."""

    text: str
    kind: ClaimKind = ClaimKind.FACT
    refs: list[str] = Field(default_factory=list)


class PresentedBlock(BaseModel):
    """A titled group of lines (e.g. 'Policy compliance')."""

    key: str
    title: str
    lines: list[PresentedLine] = Field(default_factory=list)


class PresentedRequirement(BaseModel):
    """One requirement row — ``status`` is the recorded value, invariant."""

    requirement_id: str
    requirement_text: str          # recorded policy text, unchanged
    status: str                    # SATISFIED | VIOLATED | UNKNOWN (INVARIANT)
    status_text: str               # audience wording of that same status
    confidence: float
    kind: ClaimKind
    detail: str                    # mode-adapted sentence built from recorded data
    evidence: list[str] = Field(default_factory=list)
    conflicting: list[str] = Field(default_factory=list)


class PresentedClaim(BaseModel):
    """One validation verdict — ``claim``/``status``/``evidence`` invariant."""

    claim: str                     # exact validated claim text (INVARIANT)
    rendered: str                  # mode framing that embeds ``claim`` verbatim
    status: str                    # SUPPORTED | CONTRADICTED | UNKNOWN (INVARIANT)
    method: str
    confidence: float
    explanation: str
    evidence: list[str] = Field(default_factory=list)   # INVARIANT
    kind: ClaimKind

    def refs_from_evidence(self) -> set[str]:
        """Evidence-ref ids reachable from this claim's cited evidence."""
        refs: set[str] = set()
        for line in self.evidence:
            refs.update(_refs_in(line))
        return refs


class PresentedExplanation(BaseModel):
    """The audience-adapted view of one validated report."""

    mode: PresentationMode
    audience: str
    description: str

    task_id: str
    trace_id: str
    action: str
    task: str
    headline: str

    blocks: list[PresentedBlock] = Field(default_factory=list)
    requirements: list[PresentedRequirement] = Field(default_factory=list)
    claims: list[PresentedClaim] = Field(default_factory=list)

    # --- invariants (identical in all four modes)
    evidence_index: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    unsupported_claim_rate: float = 0.0
    overall_status: str = ""
    guardrail: str = ""
    recorded: dict[str, str] = Field(default_factory=dict)

    uncertainty: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)

    # -------------------------------------------------------- invariants
    def status_digest(self) -> dict[str, str]:
        """claim text -> verification status (must match across modes)."""
        return {c.claim: c.status for c in self.claims}

    def requirement_digest(self) -> dict[str, str]:
        """requirement id -> recorded status (must match across modes)."""
        return {r.requirement_id: r.status for r in self.requirements}

    def ref_digest(self) -> frozenset[str]:
        """Every evidence reference id reachable from this explanation."""
        refs: set[str] = set(self.evidence_refs)
        for c in self.claims:
            refs.update(c.refs_from_evidence())
        for r in self.requirements:
            for line in r.evidence + r.conflicting:
                refs.update(_refs_in(line))
        for block in self.blocks:
            for line in block.lines:
                refs.update(line.refs)
        return frozenset(refs)

    # ---------------------------------------------------------- rendering
    def to_text(self) -> str:
        out: list[str] = []
        bar = "=" * 72
        out.append(bar)
        out.append(f"TRACE-XAI {self.mode.value.upper()} EXPLANATION "
                   f"{self.task_id} / {self.trace_id}")
        out.append(f"audience: {self.audience}")
        out.append(bar)
        out.append(self.headline)

        for block in self.blocks:
            out.append(f"\n-- {block.title} --")
            for line in block.lines:
                out.append(f"    [{line.kind.value}] {line.text}")

        if self.requirements:
            out.append("\n-- Requirements --")
            for r in self.requirements:
                out.append(f"    [{r.kind.value}] {r.requirement_id} "
                           f"{r.status_text}: {r.requirement_text}")
                out.append(f"        confidence={r.confidence:.2f}"
                           + (f" | evidence: {'; '.join(r.evidence)}"
                              if r.evidence else ""))

        if self.claims:
            out.append("\n-- Verified claims --")
            for c in self.claims:
                out.append(f"    [{c.kind.value}] {c.status} :: {c.claim}")
                if c.evidence:
                    out.append(f"        evidence: {'; '.join(c.evidence)}")

        out.append(f"\n-- Evidence index ({len(self.evidence_index)} entries) --")
        out.extend(f"    {e}" for e in self.evidence_index)

        out.append("\n-- Validation --")
        out.append(f"    counts: {self.counts}")
        out.append(f"    unsupported_claim_rate: {self.unsupported_claim_rate:.3f}")
        out.append(f"    overall: {self.overall_status} | "
                   f"guardrail: {self.guardrail}")

        if self.uncertainty:
            out.append("\n-- Uncertainty --")
            out.extend(f"    {u}" for u in self.uncertainty)
        if self.limitations:
            out.append("\n-- Limitations --")
            out.extend(f"    {l}" for l in self.limitations)
        out.append("\n" + "-" * 72)
        out.append("synthetic data only | wording adapts to the audience; "
                   "evidence and conclusions do not change")
        return "\n".join(out)

    def to_markdown(self) -> str:
        """Markdown rendering of the *same* items as ``to_text()``.

        Presentation-only alternative for UI panels: real headings instead
        of ``-- title --`` separators, and a legend for the epistemic tags.
        Every status, claim text, evidence reference and count is the same
        recorded value as in ``to_text()`` — this method re-renders, it never
        recomputes (asserted by tests/unit/test_presentation.py).
        """
        out: list[str] = []
        out.append(f"#### {self.mode.value.upper()} explanation — "
                   f"{self.audience}")
        out.append(self.description)
        out.append("")
        out.append(f"> {self.headline}")
        out.append("")
        out.append("**How to read the tags** — `FACT`: settled by the "
                   "recorded evidence · `INFERENCE`: derived or predicted, "
                   "not directly observed · `UNKNOWN`: not determinable "
                   "from the recorded evidence (never upgraded to a fact).")

        for block in self.blocks:
            out.append("")
            out.append(f"### {block.title}")
            for line in block.lines:
                out.append(f"- `{line.kind.value}` {line.text}")

        if self.requirements:
            out.append("")
            out.append("### Requirements")
            for r in self.requirements:
                out.append(f"- `{r.kind.value}` **{r.requirement_id}** — "
                           f"{r.status_text}: {r.requirement_text} "
                           f"(confidence {r.confidence:.2f})")
                if r.evidence:
                    out.append(f"  - evidence: {'; '.join(r.evidence)}")
                if r.conflicting:
                    out.append(f"  - conflicting: {'; '.join(r.conflicting)}")

        if self.claims:
            out.append("")
            out.append("### Verified claims")
            for c in self.claims:
                out.append(f"- `{c.status}` {c.claim}")
                if c.evidence:
                    out.append(f"  - evidence: {'; '.join(c.evidence)}")

        out.append("")
        out.append(f"### Evidence index ({len(self.evidence_index)} entries)")
        for e in self.evidence_index:
            out.append(f"- {e}")

        out.append("")
        out.append("### Validation")
        out.append(f"- counts: {self.counts}")
        out.append(f"- unsupported_claim_rate: "
                   f"{self.unsupported_claim_rate:.3f}")
        out.append(f"- overall: {self.overall_status} · guardrail: "
                   f"{self.guardrail}")

        if self.uncertainty:
            out.append("")
            out.append("### Uncertainty")
            out.extend(f"- {u}" for u in self.uncertainty)
        if self.limitations:
            out.append("")
            out.append("### Limitations")
            out.extend(f"- {l}" for l in self.limitations)

        out.append("")
        out.append("---")
        out.append("*synthetic data only · wording adapts to the audience; "
                   "evidence and conclusions do not change*")
        return "\n".join(out)


# ------------------------------------------------------------ internals
def _refs_in(text: str) -> set[str]:
    return set(re.findall(r"\b(?:E\d+|REQ-[A-Z]+-\d+|ANOM-\d{4}|CF-\d{4})\b", text))


def _claim_refs(evidence_lines: list[str]) -> set[str]:
    refs: set[str] = set()
    for line in evidence_lines:
        refs.update(_refs_in(line))
    return refs


def _action_facts(lines: list[str]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in lines:
        match = _ACTION_PAIR.match(line)
        if match:
            facts[match.group(1)] = match.group(2)
    return facts


def _outcome_parts(outcome: str) -> tuple[bool | None, str]:
    """(success?, plain-language reason) from a recorded outcome line."""
    low = outcome.lower()
    success: bool | None = None
    if low.startswith("success"):
        success = True
    elif low.startswith("rejected") or low.startswith("fail"):
        success = False
    code_match = re.search(r"reason_code=([A-Z_]+)", outcome)
    code = code_match.group(1) if code_match else None
    if code is None:
        return success, outcome
    plain = _REASON_PLAIN.get(code, code)  # unknown code: pass through
    if success is True:
        return True, plain
    if success is False:
        return False, f"{plain} (recorded code {code})"
    return None, f"{plain} (recorded code {code})"


def _kind_for_requirement(status: str) -> ClaimKind:
    return ClaimKind.UNKNOWN if status.upper() == "UNKNOWN" else ClaimKind.FACT


def _kind_for_claim(status: str) -> ClaimKind:
    # evidence-settled verdicts (even a refutation) vs. not determinable
    return ClaimKind.UNKNOWN if status == "UNKNOWN" else ClaimKind.FACT


def _evidence_for(status_obj, all_lines: list[str]) -> list[str]:
    """Rendered evidence lines that belong to one requirement assessment."""
    out: list[str] = []
    for item in status_obj.supporting_evidence + status_obj.conflicting_evidence:
        needle = item.event_id
        for line in all_lines:
            if needle in line and (item.field in line):
                out.append(line)
                break
    return out


def _requirement_status_map(report) -> dict[str, str]:
    return dict(report.requirement_satisfaction)


# Claim-verdict wording per mode (verification status, distinct from the
# requirement status table in modes.py). The recorded status itself is never
# altered — only how it is phrased for the audience.
_CLAIM_STATUS_WORD: dict[PresentationMode, dict[str, str]] = {
    PresentationMode.TECHNICAL: {
        "SUPPORTED": "SUPPORTED", "CONTRADICTED": "CONTRADICTED",
        "UNKNOWN": "UNKNOWN",
    },
    PresentationMode.BUSINESS: {
        "SUPPORTED": "backed by the records",
        "CONTRADICTED": "contradicted by the records",
        "UNKNOWN": "not verifiable from the records",
    },
    PresentationMode.NORMAL: {
        "SUPPORTED": "the records back it up",
        "CONTRADICTED": "the records disagree with it",
        "UNKNOWN": "we cannot tell from the records",
    },
    PresentationMode.ELI5: {
        "SUPPORTED": "the notes back it up",
        "CONTRADICTED": "the notes say otherwise",
        "UNKNOWN": "the notes do not tell us",
    },
}

_RISK_BY_CATEGORY = {
    "NONE": "No policy breach and no process irregularity was recorded "
            "in this run.",
    "REQUIREMENT_VIOLATION": "One or more registered policy requirements "
                             "were breached by the recorded action.",
    "TRAJECTORY_ANOMALY": "Irregularities were recorded in the step "
                          "sequence; no registered policy requirement was "
                          "breached.",
    "BOTH": "Registered policy requirement(s) were breached and irregular "
            "steps were recorded in the same run.",
}


def _analogy(report, n_violated: int, outcome_success: bool | None,
             n_unknown: int) -> str:
    """Deterministic, clearly-labelled analogy for ELI5 (never a fact claim)."""
    if n_violated:
        text = ("Analogy (not evidence): it is like a height limit on a "
                "water slide — you only ride if you are under the line, and "
                "this request was over the line.")
    elif outcome_success:
        text = ("Analogy (not evidence): it is like a teacher checking "
                "your homework against the answer key before handing it "
                "back — this time the checks passed.")
    else:
        text = ("Analogy (not evidence): it is like a vending machine "
                "turning down a note it does not recognise — the request "
                "was refused for a reason written in the log.")
    if n_unknown:
        text += (" Some notes were missing, like a page torn out of a "
                 "notebook — so we say 'we do not know' instead of "
                 "guessing.")
    return text


def _headline(report, mode: PresentationMode, facts: dict[str, str],
              outcome_success: bool | None, outcome_plain: str,
              counts: dict[str, int], rate: float, n_ok: int, n_total: int,
              n_unknown_req: int) -> str:
    action = report.action
    overall = report.validation.overall_validation_status.value
    guard = report.validation.guardrail_decision.value
    if mode is PresentationMode.TECHNICAL:
        return (f"{action} | task={report.task_id} | trace={report.trace_id} "
                f"| guardrail={guard} | claims "
                f"{counts['supported']}/{counts['contradicted']}"
                f"/{counts['unknown']} (S/C/U) "
                f"| unsupported_rate={rate:.2f} | overall={overall}")
    if mode is PresentationMode.BUSINESS:
        state = ("completed" if outcome_success is True else
                 "not completed" if outcome_success is False else
                 "outcome not determinable")
        return (f"Action {action}: {state} ({outcome_plain}). "
                f"Policy compliance: {n_ok}/{n_total} registered "
                f"requirement(s) met"
                + (f", {n_unknown_req} not assessable"
                   if n_unknown_req else "")
                + f". Explanation evidence: {overall} "
                  f"(guardrail {guard}).")
    if mode is PresentationMode.NORMAL:
        state = ("went through" if outcome_success is True else
                 "was refused" if outcome_success is False else
                 "has an outcome we cannot determine")
        return (f"The agent tried a {action.lower()} action and it {state}: "
                f"{outcome_plain}. {n_ok} of {n_total} recorded rule(s) were "
                f"followed.")
    # ELI5
    state = ("worked" if outcome_success is True else
             "did not work" if outcome_success is False else
             "gave a result we are not sure about")
    return (f"The robot tried one job ({action.lower()}) and it {state} "
            f"({outcome_plain}). {n_ok} out of {n_total} rule(s) were okay.")


def present_report(report: "FinalXAIReport",
                   mode: PresentationMode | str) -> PresentedExplanation:
    """Build the audience-adapted view of one validated report.

    Pure function of ``report``: no model calls, no recomputation of any
    verification status, count or guardrail decision.
    """
    if isinstance(mode, str):
        mode = PresentationMode(mode)

    facts = _action_facts(report.agent_action)
    outcome = facts.get("outcome", "")
    outcome_success, outcome_plain = _outcome_parts(outcome)

    v = report.validation
    counts = {
        "supported": v.supported_claim_count,
        "contradicted": v.contradicted_claim_count,
        "unknown": v.unknown_claim_count,
        "total": len(v.verifications),
    }
    rate = v.unsupported_claim_rate
    statuses = [s.upper() for s in report.requirement_satisfaction.values()]
    n_total = len(statuses)
    n_ok = sum(1 for s in statuses if s == "SATISFIED")
    n_viol = sum(1 for s in statuses if s == "VIOLATED")
    n_unknown_req = sum(1 for s in statuses if s == "UNKNOWN")

    headline = _headline(report, mode, facts, outcome_success, outcome_plain,
                         counts, rate, n_ok, n_total, n_unknown_req)

    intro = INTRO[mode]
    blocks: list[PresentedBlock] = []

    # --- 1. what the agent did (recorded observation lines, all modes)
    action_lines = [
        PresentedLine(text=line, kind=ClaimKind.FACT,
                      refs=sorted(_refs_in(line)))
        for line in report.agent_action
    ]
    if mode in (PresentationMode.NORMAL, PresentationMode.ELI5) and outcome:
        action_lines.append(PresentedLine(
            text=f"Recorded outcome: {outcome}",
            kind=ClaimKind.FACT))
    if mode is PresentationMode.ELI5:
        # clearly-labelled analogy; tagged INFERENCE so it is never read as
        # an observed fact of this trace
        action_lines.append(PresentedLine(
            text=_analogy(report, n_viol, outcome_success, n_unknown_req),
            kind=ClaimKind.INFERENCE))
    blocks.append(PresentedBlock(key="action", title=intro["action"],
                                 lines=action_lines))

    # --- 4. rationale (recorded verbatim, so no source fact is lost)
    if report.decision_rationale:
        rationale_lines: list[PresentedLine] = []
        if mode in (PresentationMode.NORMAL, PresentationMode.ELI5):
            plain = (f"The check against the recorded rules found {n_ok} of "
                     f"{n_total} rule(s) followed"
                     + (f" and {n_viol} not followed" if n_viol else "")
                     + (f"; {n_unknown_req} could not be assessed"
                        if n_unknown_req else "") + ".")
            rationale_lines.append(PresentedLine(text=plain,
                                                 kind=ClaimKind.FACT))
            label = ("Recorded rationale (kept verbatim so no detail is "
                     "lost): ")
        elif mode is PresentationMode.BUSINESS:
            label = "Recorded decision rationale: "
        else:
            label = ""
        rationale_lines.append(PresentedLine(
            text=f"{label}{report.decision_rationale}",
            kind=ClaimKind.FACT,
            refs=sorted(_refs_in(report.decision_rationale))))
        blocks.append(PresentedBlock(key="rationale",
                                     title=intro["rationale"],
                                     lines=rationale_lines))

    # --- 6/7. deviations + anomalies (+ business risk inference)
    dev_lines: list[PresentedLine] = []
    for line in report.deviations:
        dev_lines.append(PresentedLine(text=line, kind=ClaimKind.FACT,
                                       refs=sorted(_refs_in(line))))
    if mode is PresentationMode.BUSINESS:
        cat = report.deviation.category.value
        dev_lines.append(PresentedLine(
            text="Operational risk (inference from recorded findings): "
                 + _RISK_BY_CATEGORY.get(cat, cat),
            kind=ClaimKind.INFERENCE))
    if dev_lines:
        blocks.append(PresentedBlock(key="deviations",
                                     title=intro["deviations"],
                                     lines=dev_lines))

    anom_lines = [PresentedLine(text=line, kind=ClaimKind.FACT,
                                refs=sorted(_refs_in(line)))
                  for line in report.trajectory_anomalies]
    relation = report.deviation.relation.value
    if relation not in ("none",):
        anom_lines.append(PresentedLine(
            text=f"Association label (inference, never causal): anomaly-to-"
                 f"violation relation is recorded as '{relation}'.",
            kind=ClaimKind.INFERENCE))
    if anom_lines:
        blocks.append(PresentedBlock(key="anomalies",
                                     title=intro["anomalies"],
                                     lines=anom_lines))

    # --- 8. counterfactuals (always INFERENCE: they are predictions)
    cf_lines: list[PresentedLine] = []
    for line in report.counterfactuals:
        kind = ClaimKind.INFERENCE
        refs = sorted(_refs_in(line))
        if "replay performed: True" in line:
            text = (f"{line} [replay-confirmed prediction]")
        else:
            text = f"{line} [prediction only]"
        cf_lines.append(PresentedLine(text=text, kind=kind, refs=refs))
    if cf_lines:
        blocks.append(PresentedBlock(key="counterfactuals",
                                     title=intro["counterfactuals"],
                                     lines=cf_lines))

    # --- 10/11/12/13. claims (verbatim text + status, framed for audience)
    claims: list[PresentedClaim] = []
    words = _CLAIM_STATUS_WORD[mode]
    for verdict in v.verifications:
        status = verdict.verification_status.value
        word = words.get(status, status)
        if mode is PresentationMode.TECHNICAL:
            rendered = (f"[{status} | {verdict.verification_method.value} | "
                        f"confidence={verdict.confidence:.2f}] "
                        f"{verdict.claim}")
        elif mode is PresentationMode.BUSINESS:
            rendered = (f"Checked against the records — {verdict.claim} "
                        f"(verification: {word}).")
        elif mode is PresentationMode.NORMAL:
            rendered = (f"The explanation says: \"{verdict.claim}\" "
                        f"— {word}.")
        else:
            rendered = (f"We checked this sentence: \"{verdict.claim}\" "
                        f"— {word}.")
        claims.append(PresentedClaim(
            claim=verdict.claim, rendered=rendered, status=status,
            method=verdict.verification_method.value,
            confidence=verdict.confidence, explanation=verdict.explanation,
            evidence=list(verdict.evidence),
            kind=_kind_for_claim(status)))

    # --- requirement rows (status copied, wording adapted)
    requirements: list[PresentedRequirement] = []
    for assessment in report.requirements:
        status = assessment.status.value
        if mode is PresentationMode.TECHNICAL:
            detail = assessment.explanation
        elif mode is PresentationMode.BUSINESS:
            detail = f"Policy: {assessment.requirement_text}"
            if status == "VIOLATED":
                detail += (" | Business impact: the recorded action did not "
                           "meet this policy (policy breach recorded).")
            elif status == "UNKNOWN":
                detail += (" | Impact cannot be assessed: required evidence "
                           "is missing or conflicting.")
        else:
            detail = f"Rule: {assessment.requirement_text}"
        requirements.append(PresentedRequirement(
            requirement_id=assessment.requirement_id,
            requirement_text=assessment.requirement_text,
            status=status,
            status_text=status_words(mode, status),
            confidence=assessment.confidence,
            kind=_kind_for_requirement(status),
            detail=detail,
            evidence=_evidence_for(assessment, report.evidence),
            conflicting=[f"{i.event_id}.{i.field} = {i.value!r}"
                         for i in assessment.conflicting_evidence],
        ))

    blocks.append(PresentedBlock(key="claims", title=intro["claims"],
                                 lines=[PresentedLine(
                                     text=c.rendered, kind=c.kind,
                                     refs=sorted(c.refs_from_evidence()))
                                     for c in claims]))
    blocks.append(PresentedBlock(key="evidence", title=intro["evidence"],
                                 lines=[PresentedLine(text=e,
                                                      kind=ClaimKind.FACT,
                                                      refs=sorted(_refs_in(e)))
                                        for e in report.evidence]))
    if report.uncertainty:
        blocks.append(PresentedBlock(
            key="uncertainty", title=intro["uncertainty"],
            lines=[PresentedLine(text=u, kind=ClaimKind.UNKNOWN,
                                 refs=sorted(_refs_in(u)))
                   for u in report.uncertainty]))
    if report.limitations:
        blocks.append(PresentedBlock(
            key="limitations", title=intro["limitations"],
            lines=[PresentedLine(text=l, kind=ClaimKind.FACT)
                   for l in report.limitations]))

    # evidence refs used anywhere (accessible in every mode)
    refs: set[str] = set()
    for line in report.evidence:
        refs.update(_refs_in(line))
    for verdict in v.verifications:
        for line in verdict.evidence:
            refs.update(_refs_in(line))
    for assessment in report.requirements:
        for item in assessment.supporting_evidence + assessment.conflicting_evidence:
            refs.update(_refs_in(f"{item.event_id} {item.field}"))

    return PresentedExplanation(
        mode=mode,
        audience=MODE_AUDIENCE[mode],
        description=MODE_DESCRIPTION[mode],
        task_id=report.task_id, trace_id=report.trace_id,
        action=report.action, task=report.task, headline=headline,
        blocks=blocks, requirements=requirements, claims=claims,
        evidence_index=list(report.evidence),
        evidence_refs=sorted(refs),
        counts=counts, unsupported_claim_rate=rate,
        overall_status=v.overall_validation_status.value,
        guardrail=v.guardrail_decision.value,
        recorded={
            "decision_rationale": report.decision_rationale,
            "deviation_explanation": report.deviation.combined_explanation,
            "task": report.task,
        },
        uncertainty=list(report.uncertainty),
        limitations=list(report.limitations),
        provenance={
            "source": "validated FinalXAIReport",
            "source_method": report.method,
            "report_llm_used": report.llm_used,
            "presentation_generator": "deterministic_wording_v1",
            "presentation_llm_used": False,
            "synthetic_data": True,
        },
    )


def present_all(report: "FinalXAIReport") -> dict[PresentationMode,
                                                  PresentedExplanation]:
    """All four modes from one validated report (used by tests + GUI)."""
    return {mode: present_report(report, mode)
            for mode in PresentationMode}
