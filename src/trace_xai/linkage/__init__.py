"""Metric-to-trace linkage (Part 5): evidence chains for metric movements.

Connects a metric change to the exact trace events, explanation claims,
evidence references and validation statuses it can legitimately reach:

    METRIC -> OBSERVED CHANGE -> RELEVANT TRACE EVENTS
           -> EXPLANATION CLAIM -> EVIDENCE REFERENCES -> VALIDATION STATUS

The result is labelled ASSOCIATION (co-located in the same validated run) or
HYPOTHESIS (no trace evidence connects the movement). Causality is never
claimed from metric movement alone, and CSAT/retention movements are never
linked causally to explanation quality.

Usage::

    from trace_xai.linkage import MetricChange, link_metric

    link = link_metric(report, MetricChange(
        metric="unsupported_claim_rate", before=0.4, after=0.0,
        unit="ratio", basis="measured",
        related_requirement_ids=["REQ-AMT-001"]))
    print(link.to_text())
"""

from .linker import link_metric
from .models import LinkType, LinkedClaim, MetricChange, MetricTraceLink

__all__ = [
    "link_metric",
    "MetricChange",
    "MetricTraceLink",
    "LinkedClaim",
    "LinkType",
]
