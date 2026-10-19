"""Metrics: latency, cost, claim quality, CSAT — each with definition + basis.

Guarantees (research brief, Part 3):

- Latency separates agent execution from XAI analysis and end-to-end, all in
  milliseconds from ``perf_counter``; unmeasured components are UNKNOWN, not 0.
- Cost uses real provider-reported tokens; local models have no price, so
  cost is UNKNOWN unless the caller supplies documented pricing (then it is
  labelled ESTIMATED).
- Unsupported-claim rate has an exact documented formula and decomposes into
  contradiction (evidence refutes) and unverifiable (evidence insufficient);
  UNKNOWN is never counted as a hallucination.
- CSAT/retention: no real user study exists; real CSAT is NOT_COLLECTED and
  the simulated variant is deterministic, labelled and formula-documented.

Usage::

    from trace_xai import analyze_execution
    report = analyze_execution("Transfer ₹10,000 to beneficiary BOB-001.")
    metrics = report.metrics(simulate_csat=True)
    print(metrics.latency.agent_execution_ms, metrics.cost.basis)
"""

from .collector import CSATRecorder, collect_metrics, evaluate_detection
from .models import (
    ClaimDetectionMetrics,
    ClaimQualityMetrics,
    CostMetrics,
    CSATMetrics,
    InjectedClaim,
    LatencyMetrics,
    MetricsBundle,
    MetricBasis,
    ModelPricing,
    SimulatedCSAT,
    SimulatedRetention,
    TokenUsage,
)

__all__ = [
    "collect_metrics",
    "evaluate_detection",
    "CSATRecorder",
    "MetricsBundle",
    "MetricBasis",
    "LatencyMetrics",
    "CostMetrics",
    "ClaimQualityMetrics",
    "ClaimDetectionMetrics",
    "CSATMetrics",
    "SimulatedCSAT",
    "SimulatedRetention",
    "ModelPricing",
    "TokenUsage",
    "InjectedClaim",
]
