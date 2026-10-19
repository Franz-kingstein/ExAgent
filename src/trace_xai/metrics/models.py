"""Metric models: every metric carries definition, unit and basis.

Rules enforced across this package (research brief, Part 3):

1. Units and formulas are explicit on every metric (no bare numbers).
2. Each metric states its *basis* — measured, estimated, simulated or
   unknown. Unknown pricing/feedback is reported as unknown, never
   fabricated.
3. Unsupported-claim rate is decomposed: contradicted (evidence refutes the
   claim) and unknown (evidence insufficient) are reported separately, and
   UNKNOWN is explicitly **not** counted as a hallucination.
4. CSAT/retention: no real user study has been conducted. Real CSAT is
   NOT_COLLECTED (0 responses); any rating shown is a deterministic
   SIMULATED stand-in, labelled as such, with its formula and sample size
   stated. CSAT is a satisfaction proxy, distinct from factual correctness.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MetricBasis(str, Enum):
    """How a metric value came to be — required context for every number."""

    MEASURED = "measured"        # directly observed in this run
    ESTIMATED = "estimated"      # measured inputs + documented formula
    SIMULATED = "simulated"      # synthetic stand-in; no real users involved
    UNKNOWN = "unknown"          # not determinable; reported, never invented
    NOT_APPLICABLE = "not_applicable"


# --------------------------------------------------------------- latency
class LatencyMetrics(BaseModel):
    """Wall-clock latency in milliseconds, split by what was timed.

    Method: ``time.perf_counter()`` around each stage, rounded to 0.001 ms.
    Agent execution and XAI analysis are timed separately so a slow model
    call is never conflated with explanation overhead. When the agent did
    not run in this call (trace supplied directly), agent latency is
    ``None`` with basis UNKNOWN — never silently zero.
    """

    unit: str = "ms"
    method: str = ("time.perf_counter() wall-clock per stage, "
                   "rounded to 0.001 ms")
    agent_execution_ms: float | None = None
    agent_basis: MetricBasis = MetricBasis.UNKNOWN
    xai_analysis_ms: float | None = None
    xai_stage_ms: dict[str, float] = Field(default_factory=dict)
    end_to_end_ms: float | None = None
    end_to_end_basis: MetricBasis = MetricBasis.UNKNOWN
    notes: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- cost
class TokenUsage(BaseModel):
    """Real provider-reported token usage for one model call stage."""

    stage: str                    # "rationale" | "judge"
    provider: str
    model: str | None = None
    prompt_tokens: int | None = None      # provider-reported prompt eval count
    completion_tokens: int | None = None  # provider-reported generation count
    duration_ms: float | None = None      # provider-reported total duration


class ModelPricing(BaseModel):
    """Optional pricing input for cost *estimates* only.

    TRACE-XAI ships no pricing table (prices change and a local model has
    none). If a caller supplies rates, cost is computed and labelled
    ESTIMATED with the source recorded; without rates, cost is UNKNOWN.
    """

    usd_per_prompt_token: float = Field(ge=0.0)
    usd_per_completion_token: float = Field(ge=0.0)
    source: str = Field(description="Where these rates come from (documented)")


class CostMetrics(BaseModel):
    """Token usage (measured) and optional cost estimate (labelled).

    Formula for the estimate:
        cost_usd = prompt_tokens * usd_per_prompt_token
                 + completion_tokens * usd_per_completion_token
    Local models report tokens separately and have no pricing -> cost UNKNOWN.
    """

    unit: str = "USD"
    usage: list[TokenUsage] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    pricing_known: bool = False
    pricing_source: str | None = None
    estimated_cost_usd: float | None = None
    basis: MetricBasis = MetricBasis.UNKNOWN
    formula: str = ("estimated_cost_usd = prompt_tokens * "
                    "usd_per_prompt_token + completion_tokens * "
                    "usd_per_completion_token")
    notes: list[str] = Field(default_factory=list)


# -------------------------------------------------- claim quality metrics
class ClaimQualityMetrics(BaseModel):
    """Unsupported-claim / hallucination-oriented metrics, fully decomposed.

    Exact formulas (documented here so the numbers are reproducible):
        unsupported_claim_rate = (contradicted + unknown) / total_claims
        contradiction_rate     = contradicted / total_claims
        unverifiable_rate      = unknown / total_claims
    UNKNOWN means the recorded evidence is insufficient to decide — it is
    **not** a hallucination and is never reported as one.
    """

    total_claims: int = 0
    supported_count: int = 0
    contradicted_count: int = 0
    unknown_count: int = 0
    unsupported_claim_rate: float = 0.0
    contradiction_rate: float = 0.0
    unverifiable_rate: float = 0.0
    formula: str = ("unsupported_claim_rate = (contradicted + unknown) / "
                    "total_claims; contradiction_rate = contradicted / "
                    "total_claims; unverifiable_rate = unknown / total_claims")
    notes: list[str] = Field(default_factory=list)


class InjectedClaim(BaseModel):
    """A claim with ground truth known by construction (synthetic setup)."""

    text: str
    ground_truth: str = Field(
        description="SUPPORTED or CONTRADICTED, known by construction")
    origin: str = "synthetic_ground_truth"


class ClaimDetectionMetrics(BaseModel):
    """Detection quality against synthetic ground truth / injected claims.

    Exact formulas:
        false_acceptance_rate = wrong_accepted / wrong_claims
        detection_rate        = wrong_detected / wrong_claims
    Ground truth is known by construction (synthetic data + injected wrong
    claims); the LLM judge is never the ground truth.
    """

    total_claims: int = 0
    wrong_claims: int = 0
    correct_claims: int = 0
    wrong_detected: int = 0
    wrong_accepted: int = Field(
        ge=0, description="wrong claims judged SUPPORTED (false acceptances)")
    correct_supported: int = 0
    missed_correct: int = 0
    false_acceptance_rate: float | None = None
    detection_rate: float | None = None
    formula: str = ("false_acceptance_rate = wrong_accepted / wrong_claims; "
                    "detection_rate = wrong_detected / wrong_claims")
    notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ CSAT
CSAT_SCALE_DEFINITION = ("1 = very dissatisfied, 2 = dissatisfied, "
                         "3 = neutral, 4 = satisfied, 5 = very satisfied")


class SimulatedCSAT(BaseModel):
    """Deterministic simulated CSAT — a demonstration stand-in only.

    Formula (deterministic, no randomness):
        mean = clamp(5.0
                     - 2.0 * [guardrail == BLOCK]
                     - 1.0 * [guardrail == FLAG]
                     - 2.0 * unsupported_claim_rate,
                     1.0, 5.0)
    The simulated cohort size is a fixed synthetic parameter, not an
    observed sample.
    """

    label: str = "SIMULATED — not human feedback"
    formula: str = ("simulated_mean = clamp(5.0 - 2.0*[guardrail==BLOCK] - "
                    "1.0*[guardrail==FLAG] - 2.0*unsupported_claim_rate, "
                    "1.0, 5.0)")
    simulated_responses: int = Field(
        default=30, description="fixed synthetic cohort size (not observed)")
    simulated_mean_rating: float
    inputs: dict = Field(default_factory=dict)


class SimulatedRetention(BaseModel):
    """Simulated retention — labelled, never an observed user behaviour.

    Formula (deterministic):
        retention = clamp(0.2 + 0.15 * (simulated_csat - 1), 0.0, 1.0)
    """

    label: str = "SIMULATED — not observed user behaviour"
    formula: str = ("retention = clamp(0.2 + 0.15 * (simulated_csat - 1), "
                    "0.0, 1.0)")
    estimated_retention_rate: float
    inputs: dict = Field(default_factory=dict)


class CSATMetrics(BaseModel):
    """Satisfaction metric: real CSAT is not collected; simulation labelled.

    Real CSAT would require a user study; none has been conducted, so
    ``responses`` is 0 and ``mean_rating`` is None (NOT collected — no
    fabricated feedback). CSAT is a *satisfaction* proxy and is distinct
    from factual correctness (claim quality is reported separately).
    """

    scale_min: int = 1
    scale_max: int = 5
    scale_definition: str = CSAT_SCALE_DEFINITION
    responses: int = 0
    mean_rating: float | None = None
    basis: MetricBasis = MetricBasis.NOT_APPLICABLE
    status: str = "NOT_COLLECTED"
    note: str = ("no real user study has been conducted; ratings are not "
                 "fabricated. A deterministic SIMULATED CSAT may be attached "
                 "for demonstration and is labelled as simulated.")
    simulated: SimulatedCSAT | None = None
    simulated_retention: SimulatedRetention | None = None


# ---------------------------------------------------------------- bundle
class MetricsBundle(BaseModel):
    """All metrics for one report, each with definition/unit/basis."""

    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    cost: CostMetrics = Field(default_factory=CostMetrics)
    claim_quality: ClaimQualityMetrics = Field(default_factory=ClaimQualityMetrics)
    csat: CSATMetrics = Field(default_factory=CSATMetrics)
