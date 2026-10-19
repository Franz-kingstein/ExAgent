"""Build metric views from validated artifacts — no recomputation of verdicts.

``collect_metrics(report, ...)`` derives every metric from data already
recorded on the report (stage timings, provider usage, validation verdicts).
It is a pure function: calling it twice yields identical numbers, and it
never re-runs any analysis stage.

Also here:

- ``evaluate_detection`` — detection quality against synthetic ground truth
  and injected wrong claims (used by the Phase 6 experiment, Part 6B). The
  ground truth is known by construction; an LLM judge is never the ground
  truth.
- ``CSATRecorder`` — an honest satisfaction sink. Real ratings are only
  produced when a person explicitly records one; the simulated variant is a
  separate, clearly-labelled deterministic function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..pipeline import FinalXAIReport
    from ..validate.models import ClaimVerification

__all__ = [
    "collect_metrics",
    "evaluate_detection",
    "CSATRecorder",
    "MetricsBundle",
    "MetricBasis",
]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------- latency
def _latency(report: "FinalXAIReport") -> LatencyMetrics:
    stages = dict(report.timings_ms)
    xai_total = round(sum(stages.values()), 3) if stages else None
    agent_ms = report.agent_execution_ms

    notes: list[str] = []
    if agent_ms is None:
        notes.append("agent execution was not timed in this call (trace was "
                     "analyzed directly); agent latency is UNKNOWN, not zero")
    notes.append("XAI analysis time = sum of the five analysis stages "
                 "(decision, deviation, counterfactual, rationale, "
                 "validation); presentation wording time is excluded")

    if agent_ms is not None and xai_total is not None:
        end_to_end: float | None = round(agent_ms + xai_total, 3)
        end_basis = MetricBasis.MEASURED
    else:
        end_to_end = None
        end_basis = MetricBasis.UNKNOWN

    return LatencyMetrics(
        agent_execution_ms=agent_ms,
        agent_basis=(MetricBasis.MEASURED if agent_ms is not None
                     else MetricBasis.UNKNOWN),
        xai_analysis_ms=xai_total,
        xai_stage_ms=stages,
        end_to_end_ms=end_to_end,
        end_to_end_basis=end_basis,
        notes=notes,
    )


# ------------------------------------------------------------------ cost
def _cost(report: "FinalXAIReport",
          pricing: ModelPricing | None) -> CostMetrics:
    usage = list(report.model_usage)
    prompt_total = sum(u.prompt_tokens or 0 for u in usage)
    completion_total = sum(u.completion_tokens or 0 for u in usage)

    notes: list[str] = []
    if not usage:
        notes.append("no model was consulted for this report (deterministic "
                     "fallback); token usage is 0, so model cost is 0.00 USD "
                     "by construction, not by measurement of a bill")
    if any(u.provider.startswith("ollama") for u in usage):
        notes.append("local model: tokens are provider-reported and shown "
                     "separately; a local model has no per-token price, so "
                     "cost stays UNKNOWN unless pricing is supplied by the "
                     "caller")

    if pricing is not None:
        estimate = round(prompt_total * pricing.usd_per_prompt_token
                         + completion_total * pricing.usd_per_completion_token, 8)
        return CostMetrics(
            usage=usage, total_prompt_tokens=prompt_total,
            total_completion_tokens=completion_total,
            pricing_known=True, pricing_source=pricing.source,
            estimated_cost_usd=estimate, basis=MetricBasis.ESTIMATED,
            notes=notes,
        )
    if not usage:
        return CostMetrics(
            usage=usage, total_prompt_tokens=0, total_completion_tokens=0,
            pricing_known=False, pricing_source=None, estimated_cost_usd=0.0,
            basis=MetricBasis.MEASURED, notes=notes,
        )
    return CostMetrics(
        usage=usage, total_prompt_tokens=prompt_total,
        total_completion_tokens=completion_total,
        pricing_known=False, pricing_source=None, estimated_cost_usd=None,
        basis=MetricBasis.MEASURED, notes=notes,
    )


# --------------------------------------------------------- claim quality
def _claim_quality(report: "FinalXAIReport") -> ClaimQualityMetrics:
    v = report.validation
    total = len(v.verifications)
    unsupported = ((v.contradicted_claim_count + v.unknown_claim_count) / total
                   if total else 0.0)
    return ClaimQualityMetrics(
        total_claims=total,
        supported_count=v.supported_claim_count,
        contradicted_count=v.contradicted_claim_count,
        unknown_count=v.unknown_claim_count,
        unsupported_claim_rate=round(unsupported, 4),
        contradiction_rate=round(v.contradicted_claim_count / total, 4)
        if total else 0.0,
        unverifiable_rate=round(v.unknown_claim_count / total, 4)
        if total else 0.0,
        notes=["contradicted (evidence refutes the claim) and unknown "
               "(evidence insufficient) are reported separately",
               "UNKNOWN is not a hallucination: an unverifiable claim is "
               "disclosed as unverifiable, not asserted falsely"],
    )


# ------------------------------------------------------------------ CSAT
def simulated_csat(report: "FinalXAIReport",
                   responses: int = 30) -> CSATMetrics:
    """Deterministic simulated CSAT + retention (labelled, never real).

    Pure function of the report's guardrail decision and unsupported-claim
    rate; no randomness, so results are reproducible across runs.
    """
    from ..validate.models import GuardrailDecision

    v = report.validation
    unsupported = _claim_quality(report).unsupported_claim_rate
    penalty = (2.0 if v.guardrail_decision is GuardrailDecision.BLOCK
               else 1.0 if v.guardrail_decision is GuardrailDecision.FLAG
               else 0.0)
    mean = round(_clamp(5.0 - penalty - 2.0 * unsupported, 1.0, 5.0), 2)
    retention = round(_clamp(0.2 + 0.15 * (mean - 1), 0.0, 1.0), 4)
    return CSATMetrics(
        responses=0, mean_rating=None,
        basis=MetricBasis.NOT_APPLICABLE, status="NOT_COLLECTED",
        simulated=SimulatedCSAT(
            simulated_responses=responses, simulated_mean_rating=mean,
            inputs={"guardrail": v.guardrail_decision.value,
                    "unsupported_claim_rate": unsupported},
        ),
        simulated_retention=SimulatedRetention(
            estimated_retention_rate=retention,
            inputs={"simulated_csat": mean},
        ),
    )


class CSATRecorder:
    """Records *real* satisfaction ratings — only what a person entered.

    There is no default dataset and no inferred feedback: an empty recorder
    reports NOT_COLLECTED with 0 responses. The simulated view is a separate
    explicit call and is always labelled simulated.
    """

    SCALE = (1, 5)

    def __init__(self) -> None:
        self._ratings: list[int] = []

    def record(self, rating: int) -> None:
        low, high = self.SCALE
        if not isinstance(rating, int) or not low <= rating <= high:
            raise ValueError(f"rating must be an integer in [{low}, {high}]")
        self._ratings.append(rating)

    @property
    def response_count(self) -> int:
        return len(self._ratings)

    def metrics(self, report: "FinalXAIReport | None" = None,
                simulate: bool = False, responses: int = 30) -> CSATMetrics:
        if self._ratings:
            mean = round(sum(self._ratings) / len(self._ratings), 2)
            real = CSATMetrics(
                responses=len(self._ratings), mean_rating=mean,
                basis=MetricBasis.MEASURED, status="COLLECTED",
                note="ratings recorded explicitly by people in this session; "
                     "a convenience log, not a controlled user study",
            )
        else:
            real = CSATMetrics(
                responses=0, mean_rating=None,
                basis=MetricBasis.NOT_APPLICABLE, status="NOT_COLLECTED",
            )
        if simulate and report is not None:
            sim = simulated_csat(report, responses=responses)
            real.simulated = sim.simulated
            real.simulated_retention = sim.simulated_retention
        return real


# ---------------------------------------------------------------- bundle
def collect_metrics(report: "FinalXAIReport",
                    pricing: ModelPricing | None = None,
                    simulate_csat: bool = False,
                    csat: CSATRecorder | None = None,
                    ) -> MetricsBundle:
    """Derive the full metric view for one validated report.

    Pure function of recorded data — safe to call repeatedly (the GUI and
    the report exporter both call it on demand).
    """
    if csat is not None:
        csat_metrics = csat.metrics(report, simulate=simulate_csat)
    elif simulate_csat:
        csat_metrics = simulated_csat(report)
    else:
        csat_metrics = CSATMetrics()

    return MetricsBundle(
        latency=_latency(report),
        cost=_cost(report, pricing),
        claim_quality=_claim_quality(report),
        csat=csat_metrics,
    )


# ------------------------------------------------ detection (experiments)
def evaluate_detection(
    verdicts: list["ClaimVerification"],
    ground_truth: list[InjectedClaim],
) -> ClaimDetectionMetrics:
    """Detection quality against known-by-construction ground truth.

    Every injected/synthetic claim's text is matched to its verdict; a claim
    counts as *detected* when its verdict is not SUPPORTED (CONTRADICTED or
    UNKNOWN both stop a wrong claim from being presented as verified), and
    as a *false acceptance* when a wrong claim is judged SUPPORTED.
    """
    by_text = {v.claim: v.verification_status.value for v in verdicts}
    wrong = wrong_detected = wrong_accepted = 0
    correct = correct_supported = missed_correct = 0
    unmatched = 0

    for item in ground_truth:
        status = by_text.get(item.text)
        if status is None:
            unmatched += 1
            continue
        is_wrong = item.ground_truth == "CONTRADICTED"
        if is_wrong:
            wrong += 1
            if status == "SUPPORTED":
                wrong_accepted += 1
            else:
                wrong_detected += 1
        else:
            correct += 1
            if status == "SUPPORTED":
                correct_supported += 1
            else:
                missed_correct += 1

    notes = [
        f"{unmatched} ground-truth claim(s) did not appear in the verdict "
        f"list (not scored)" if unmatched else "all ground-truth claims were "
        "matched to a verdict",
        "ground truth is synthetic and known by construction; no LLM judge "
        "serves as ground truth",
        "CONTRADICTED and UNKNOWN both count as detected (neither is "
        "presented as verified); their split is visible in the raw verdicts",
    ]
    return ClaimDetectionMetrics(
        total_claims=len(ground_truth), wrong_claims=wrong,
        correct_claims=correct, wrong_detected=wrong_detected,
        wrong_accepted=wrong_accepted, correct_supported=correct_supported,
        missed_correct=missed_correct,
        false_acceptance_rate=(round(wrong_accepted / wrong, 4)
                               if wrong else None),
        detection_rate=(round(wrong_detected / wrong, 4) if wrong else None),
        notes=notes,
    )
