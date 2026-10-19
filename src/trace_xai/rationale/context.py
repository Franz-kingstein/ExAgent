"""Rationale context: the explicit evidence packet fed to the SLM.

The context is assembled *deterministically* from the trace and the Q1/Q2/Q3
explanations. Every fact the model needs is spelled out; anything not
recorded is rendered as an explicit ``UNKNOWN (not recorded)`` line, so the
model never has to infer missing evidence and never has to guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.trace import ExecutionTrace
from ..explain.counterfactual import CounterfactualExplainer
from ..explain.decision import DecisionExplainer
from ..explain.deviation import DeviationExplainer
from ..explain.matcher import extract_facts
from ..explain.models import (
    CounterfactualExplanation,
    DecisionExplanation,
    DeviationExplanation,
    EvidenceItem,
    RequirementStatus,
)

UNKNOWN = "UNKNOWN (not recorded)"


@dataclass
class EvidenceEntry:
    """One citable evidence item, assigned a stable E-number."""

    ref: str  # "E1", "E2", ...
    event_id: str
    field: str
    value: Any
    source: str  # which explanation layer cited it

    def ref_line(self) -> str:
        return f"{self.ref} = {self.event_id}.{self.field} = {self.value!r} (from {self.source})"


@dataclass
class RationaleContext:
    """Fully explicit, deterministic input for the rationale generator."""

    task_id: str
    trace_id: str
    task_text: str
    agent_id: str
    synthetic: bool

    user_requirement: list[str] = field(default_factory=list)
    agent_action: list[str] = field(default_factory=list)
    requirement_matching: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    trajectory_anomalies: list[str] = field(default_factory=list)
    counterfactuals: list[str] = field(default_factory=list)
    evidence: list[EvidenceEntry] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    # ---------------------------------------------------------- citable ids
    def allowed_refs(self) -> set[str]:
        """Every token a claim may cite: E-numbers + domain ids in context."""
        refs = {e.ref for e in self.evidence}
        pattern = re.compile(r"\b(?:REQ-[A-Z]+-\d+|ANOM-\d{4}|CF-\d{4})\b")
        for line in self._all_lines():
            refs.update(pattern.findall(line))
        return refs

    def _all_lines(self) -> list[str]:
        return (self.user_requirement + self.agent_action
                + self.requirement_matching + self.deviations
                + self.trajectory_anomalies + self.counterfactuals)

    # -------------------------------------------------------------- render
    def render(self) -> str:
        """Deterministic plain-text packet for the prompt."""
        out: list[str] = []

        def section(title: str, lines: list[str]) -> None:
            out.append(f"[{title}]")
            out.extend(f"  - {line}" for line in lines or [UNKNOWN])
            out.append("")

        section("1. USER REQUIREMENT", self.user_requirement)
        section("2. AGENT ACTION", self.agent_action)
        section("3. REQUIREMENT MATCHING (Q1)", self.requirement_matching)
        section("4. DEVIATIONS (Q2)", self.deviations)
        section("5. TRAJECTORY ANOMALIES (Q2)", self.trajectory_anomalies)
        section("6. COUNTERFACTUALS (Q3)", self.counterfactuals)
        section("7. EVIDENCE INDEX", [e.ref_line() for e in self.evidence])
        section("8. UNCERTAINTY", self.uncertainty)
        section("9. MISSING INFORMATION", self.missing)
        out.append("[10. PROVENANCE]")
        for key in sorted(self.provenance):
            out.append(f"  - {key}: {self.provenance[key]}")
        out.append("")
        out.append(f"[ALLOWED CITATION REFS] {' '.join(sorted(self.allowed_refs()))}")
        return "\n".join(out)


def build_context(
    trace: ExecutionTrace,
    decision: DecisionExplanation | None = None,
    deviation: DeviationExplanation | None = None,
    counterfactual: CounterfactualExplanation | None = None,
) -> RationaleContext:
    """Assemble the explicit evidence packet (computing any missing layer)."""
    decision = decision or DecisionExplainer().explain(trace)
    deviation = deviation or DeviationExplainer().explain(trace)
    counterfactual = counterfactual or CounterfactualExplainer().explain(trace)

    facts = extract_facts(trace)
    ctx = RationaleContext(
        task_id=trace.task_id,
        trace_id=trace.trace_id,
        task_text=trace.task_text,
        agent_id=trace.agent_id,
        synthetic=trace.synthetic,
    )

    evidence_pool: list[tuple[str, EvidenceItem]] = []

    # ------------------------------------------------- 1. user requirement
    for assessment in decision.requirements:
        ctx.user_requirement.append(
            f"{assessment.requirement_id}: \"{assessment.requirement_text}\""
        )

    # ------------------------------------------------------ 2. agent action
    ctx.agent_action.append(f"task: {trace.task_text}")
    if facts.invocation is None:
        ctx.agent_action.append("action: UNKNOWN (no tool invocation in trace)")
    else:
        ctx.agent_action.append(
            f"action: {facts.action or 'UNKNOWN'} via tool '{facts.invocation.tool_name}'"
        )
        ctx.agent_action.append(f"amount_inr: {facts.amount_inr if facts.amount_inr is not None else UNKNOWN}")
        ctx.agent_action.append(f"beneficiary_id: {facts.beneficiary_id or UNKNOWN}")
        ctx.agent_action.append(f"account_id: {facts.account_id or UNKNOWN}")
    success, code = trace.final_outcome
    ctx.agent_action.append(
        f"outcome: {'success' if success else 'rejected' if success is False else UNKNOWN}"
        + (f" (reason_code={code})" if code else "")
    )

    # ------------------------------------------- 3. requirement matching Q1
    for assessment in decision.requirements:
        ctx.requirement_matching.append(
            f"{assessment.requirement_id}: {assessment.status.value} "
            f"(confidence {assessment.confidence:.2f}) — {assessment.explanation}"
        )
        evidence_pool.extend(("decision", item) for item in assessment.supporting_evidence)
        evidence_pool.extend(("decision", item) for item in assessment.conflicting_evidence)

    # ---------------------------------------------------------- 4. deviations
    ctx.deviations.append(f"category: {deviation.category.value}")
    ctx.deviations.append(
        f"requirement violations: "
        + (", ".join(a.requirement_id for a in deviation.requirement_violations) or "none")
    )
    if deviation.unknown_requirement_ids:
        ctx.deviations.append(
            "unknown requirement ids: " + ", ".join(deviation.unknown_requirement_ids)
        )
    ctx.deviations.append(f"relation anomalies↔violations: {deviation.relation.value} (never causal)")
    ctx.deviations.append(deviation.combined_explanation)

    # -------------------------------------------------- 5. trajectory anomalies
    for anomaly in deviation.anomalies:
        ctx.trajectory_anomalies.append(
            f"{anomaly.anomaly_id} {anomaly.anomaly_type.value} "
            f"(severity {anomaly.severity.value}, confidence {anomaly.confidence:.2f}): "
            f"{anomaly.explanation}"
        )
        evidence_pool.extend(("deviation", item) for item in anomaly.evidence)

    # ------------------------------------------------------ 6. counterfactuals
    for cf in counterfactual.counterfactuals:
        replay = (
            f"replay performed: {cf.replay.execution_success} "
            f"(txn {cf.replay.transaction_id}, {cf.replay.reason_code})"
            if cf.replay is not None and cf.replay.performed
            else "replay NOT performed (prediction only)"
        )
        predicted = (
            "success" if cf.predicted_outcome.success
            else cf.predicted_outcome.reason_code or UNKNOWN
        )
        ctx.counterfactuals.append(
            f"{cf.cf_id}: {cf.changed_condition}; predicted outcome: {predicted} "
            f"(basis {cf.predicted_outcome.basis}); {replay}; "
            f"impact: {cf.impact.value}; uncertainty: {cf.uncertainty:.2f}"
        )
        evidence_pool.extend(("counterfactual", item) for item in cf.evidence)
    for note in counterfactual.notes:
        ctx.counterfactuals.append(f"note: {note}")

    # ---------------------------------------------------------- 7. evidence
    seen: set[tuple[str, str]] = set()
    counter = 0
    for source, item in evidence_pool:
        key = (item.event_id, item.field)
        if key in seen:
            continue
        seen.add(key)
        counter += 1
        ctx.evidence.append(
            EvidenceEntry(ref=f"E{counter}", event_id=item.event_id,
                          field=item.field, value=item.value, source=source)
        )

    # --------------------------------------------------------- 8. uncertainty
    for assessment in decision.requirements:
        if assessment.status is RequirementStatus.UNKNOWN:
            ctx.uncertainty.append(
                f"{assessment.requirement_id}: UNKNOWN — no evidence decides it"
            )
        else:
            ctx.uncertainty.append(
                f"{assessment.requirement_id}: {assessment.status.value} "
                f"with confidence {assessment.confidence:.2f}"
            )
    for cf in counterfactual.counterfactuals:
        ctx.uncertainty.append(f"{cf.cf_id}: uncertainty {cf.uncertainty:.2f}")
    if not counterfactual.counterfactuals:
        ctx.uncertainty.append("no counterfactuals generated (nothing to evaluate)")

    # ----------------------------------------------------------- 9. missing
    if facts.invocation is not None and not facts.invocation.state_before:
        ctx.missing.append("pre-action state (balances/limits/verification): UNKNOWN")
    if facts.account_id is None:
        ctx.missing.append("account id: UNKNOWN (not present in tool input/output)")
    for assessment in decision.requirements:
        if assessment.status is RequirementStatus.UNKNOWN:
            ctx.missing.append(f"{assessment.requirement_id} operands: incomplete evidence")
    if not counterfactual.counterfactuals:
        ctx.missing.append("counterfactuals: none available for this trace")

    # -------------------------------------------------------- 10. provenance
    ctx.provenance = {
        "synthetic_data": trace.synthetic,
        "agent_id": trace.agent_id,
        "decision_method": decision.method,
        "deviation_method": deviation.method,
        "counterfactual_method": counterfactual.method,
        "explain_layers_used_llm": False,
        "trace_id": trace.trace_id,
    }
    return ctx
