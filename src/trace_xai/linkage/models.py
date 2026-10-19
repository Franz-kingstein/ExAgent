"""Metric-to-trace linkage models (Part 5).

The chain built here is explicit and auditable:

    METRIC -> OBSERVED CHANGE -> RELEVANT TRACE EVENTS
           -> EXPLANATION CLAIM -> EVIDENCE REFERENCES -> VALIDATION STATUS

Honesty rules:

- The chain records *association*, never causation. A metric change and a
  trace event co-occurring in the same validated run is labelled
  ASSOCIATION; when no trace evidence links the change, it is labelled
  HYPOTHESIS and stated as such.
- CSAT/retention changes are never linked causally to explanation quality —
  those links are always HYPOTHESIS (a satisfaction proxy moving is not
  evidence that any explanation claim is right or wrong).
- Explanation claim texts, evidence references and validation statuses are
  copied verbatim from the validated report; the linker never recomputes or
  re-verifies anything.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LinkType(str, Enum):
    ASSOCIATION = "ASSOCIATION"   # co-located in the same validated trace
    HYPOTHESIS = "HYPOTHESIS"     # no trace evidence links the change


class MetricChange(BaseModel):
    """A metric movement the caller wants traced to evidence."""

    metric: str = Field(description="metric name, e.g. unsupported_claim_rate")
    before: float | None = None
    after: float | None = None
    unit: str = ""
    basis: str = Field(
        default="unknown",
        description="measured | estimated | simulated | unknown — carried "
                    "through from the metrics layer",
    )
    related_requirement_ids: list[str] = Field(
        default_factory=list,
        description="optional explicit focus; empty = unlinked observation",
    )


class LinkedClaim(BaseModel):
    """One explanation claim reached by the chain, with its verdict."""

    claim: str                     # verbatim validated claim text
    validation_status: str         # SUPPORTED | CONTRADICTED | UNKNOWN
    verification_method: str
    confidence: float
    evidence_references: list[str] = Field(default_factory=list)


class MetricTraceLink(BaseModel):
    """The full chain for one metric change — every step named."""

    # 1. METRIC
    metric: str
    metric_before: float | None = None
    metric_after: float | None = None
    metric_unit: str = ""
    metric_basis: str = "unknown"
    # 2. OBSERVED CHANGE
    observed_change: str = Field(
        description="plain description of the movement (or 'no movement')")
    # 3. RELEVANT TRACE EVENTS
    relevant_trace_events: list[str] = Field(
        default_factory=list,
        description="evidence-index lines for events the chain reaches")
    # 4-6. claims with their refs and verdicts
    linked_claims: list[LinkedClaim] = Field(default_factory=list)

    link_type: LinkType
    requirements_in_scope: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    method: str = "metric_trace_linkage_v1"

    # ---------------------------------------------------------- rendering
    def to_text(self) -> str:
        lines = [
            "=" * 72,
            "METRIC -> TRACE LINKAGE",
            "=" * 72,
            f"[1] METRIC: {self.metric} "
            f"({self.metric_before} -> {self.metric_after} "
            f"{self.metric_unit}, basis={self.metric_basis})",
            f"[2] OBSERVED CHANGE: {self.observed_change}",
            f"[3] RELEVANT TRACE EVENTS "
            f"({len(self.relevant_trace_events)}):",
        ]
        lines.extend(f"      {e}" for e in self.relevant_trace_events
                     or ["(none reachable)"])
        lines.append(f"[4] EXPLANATION CLAIMS ({len(self.linked_claims)}):")
        for c in self.linked_claims:
            lines.append(f"      claim: {c.claim}")
            lines.append(f"        evidence refs: "
                         f"{', '.join(c.evidence_references) or '(none)'}")
            lines.append(f"        validation: {c.validation_status} "
                         f"({c.verification_method}, "
                         f"confidence={c.confidence:.2f})")
        lines.append(f"link type: {self.link_type.value}")
        if self.requirements_in_scope:
            lines.append(f"requirements in scope: "
                         f"{', '.join(self.requirements_in_scope)}")
        if self.limitations:
            lines.append("limitations:")
            lines.extend(f"  - {l}" for l in self.limitations)
        if self.notes:
            lines.append("notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        lines.append("-" * 72)
        lines.append("association only — no causal relationship is claimed "
                     "from metric movement alone")
        return "\n".join(lines)
