"""Link metric movements to trace evidence — association, never causation.

``link_metric(report, change)`` walks the explicit chain:

    METRIC -> OBSERVED CHANGE -> RELEVANT TRACE EVENTS
           -> EXPLANATION CLAIM -> EVIDENCE REFERENCES -> VALIDATION STATUS

Scoping is deterministic: the requirements named in
``MetricChange.related_requirement_ids`` (or all requirements when none are
given) define which evidence events are relevant; their assessments point to
the exact events, and claims citing those events become the linked claims.

Labels:

- ASSOCIATION — the metric movement and the linked claims live in the same
  validated run and share trace events. Co-location, not causation.
- HYPOTHESIS — no trace evidence connects the movement (unscoped metric, or
  a satisfaction metric, which is never evidence about factual claims).

CSAT/retention movements are always HYPOTHESIS: a satisfaction proxy moving
is not evidence that any explanation claim is right or wrong.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .models import LinkType, LinkedClaim, MetricChange, MetricTraceLink

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..pipeline import FinalXAIReport

__all__ = ["link_metric"]

_SATISFACTION_METRICS = ("csat", "retention", "satisfaction")
_REQ_ID = re.compile(r"\bREQ-[A-Z]+-\d+\b")


def _change_text(change: MetricChange) -> str:
    if change.before is None and change.after is None:
        return "no before/after values supplied; movement not described"
    if change.before is None:
        return (f"only an after value is known ({change.after} "
                f"{change.unit}); no baseline, so no movement is claimed"
                .rstrip())
    if change.after is None:
        return (f"only a before value is known ({change.before} "
                f"{change.unit}); no later reading, so no movement is "
                f"claimed")
    if change.before == change.after:
        return f"no movement ({change.before} {change.unit})"
    direction = ("increased" if change.after > change.before
                 else "decreased")
    return (f"{direction} from {change.before} to {change.after} "
            f"{change.unit}".rstrip())


def link_metric(report: "FinalXAIReport",
                change: MetricChange) -> MetricTraceLink:
    """Build the metric->trace chain for one metric movement.

    Pure function of the validated report and the supplied change: no
    re-verification, no new evidence, no causal inference.
    """
    metric_low = change.metric.lower()
    is_satisfaction = any(s in metric_low for s in _SATISFACTION_METRICS)

    # ---- scope: which requirements does this movement relate to?
    scope = list(change.related_requirement_ids)
    if scope:
        known = {a.requirement_id for a in report.requirements}
        unknown = [r for r in scope if r not in known]
    else:
        known = {a.requirement_id for a in report.requirements}
        unknown = []

    limitations: list[str] = []
    notes: list[str] = [
        "chain built by co-location within one validated run: the metric "
        "movement and the linked claims share trace events, which is an "
        "association, not a causal relationship",
        "claim texts, evidence references and validation statuses are "
        "copied verbatim from the validated report",
    ]

    # ---- satisfaction metrics: never evidence about factual claims
    if is_satisfaction:
        limitations.append(
            "CSAT/retention is a satisfaction proxy; its movement is never "
            "linked causally to explanation quality or factual correctness")
        return MetricTraceLink(
            metric=change.metric,
            metric_before=change.before, metric_after=change.after,
            metric_unit=change.unit, metric_basis=change.basis,
            observed_change=_change_text(change),
            relevant_trace_events=[],
            linked_claims=[],
            link_type=LinkType.HYPOTHESIS,
            requirements_in_scope=scope,
            limitations=limitations + [
                "no trace evidence connects this movement to any specific "
                "claim; treat the relation as a hypothesis requiring a "
                "controlled study"],
            notes=notes,
        )

    # ---- resolve scope to requirement assessments
    if scope:
        assessments = [a for a in report.requirements
                       if a.requirement_id in scope]
        if unknown:
            limitations.append(
                f"requirement(s) {', '.join(unknown)} are not part of this "
                f"report; chain limited to known requirements")
    else:
        assessments = list(report.requirements)
        limitations.append(
            "no explicit requirement scope supplied; the whole report was "
            "used, so the association is broad — prefer naming "
            "related_requirement_ids for a tight chain")

    # ---- relevant trace events: the evidence those assessments rest on.
    # With an explicit scope, matching is field-level (event_id.field), so
    # the chain stays tight; unscoped runs match whole events.
    relevant: list[str] = []
    event_ids: set[str] = set()
    field_keys: set[str] = set()
    for assessment in assessments:
        for item in (assessment.supporting_evidence
                     + assessment.conflicting_evidence):
            event_ids.add(item.event_id)
            field_keys.add(f"{item.event_id}.{item.field}")
    for line in report.evidence:
        if (scope and any(key in line for key in field_keys)) or (
                not scope and any(eid in line for eid in event_ids)):
            relevant.append(line)

    # ---- linked claims: validated verdicts citing those events/fields
    linked: list[LinkedClaim] = []
    for verdict in report.validation.verifications:
        if scope:
            refs = [line for line in verdict.evidence
                    if any(key in line for key in field_keys)
                    or any(rid in line for rid in scope)]
        else:
            refs = [line for line in verdict.evidence
                    if any(eid in line for eid in event_ids)]
        if refs or (not scope and not event_ids):
            linked.append(LinkedClaim(
                claim=verdict.claim,
                validation_status=verdict.verification_status.value,
                verification_method=verdict.verification_method.value,
                confidence=verdict.confidence,
                evidence_references=refs,
            ))

    if not relevant and not linked:
        limitations.append(
            "no trace events or claims in this report connect to the "
            "scoped requirements; the movement stays an unlinked "
            "observation (hypothesis)")

    return MetricTraceLink(
        metric=change.metric,
        metric_before=change.before, metric_after=change.after,
        metric_unit=change.unit, metric_basis=change.basis,
        observed_change=_change_text(change),
        relevant_trace_events=relevant,
        linked_claims=linked,
        link_type=(LinkType.ASSOCIATION if (relevant or linked)
                   else LinkType.HYPOTHESIS),
        requirements_in_scope=sorted(scope) if scope
        else sorted({a.requirement_id for a in assessments}),
        limitations=limitations,
        notes=notes,
    )
