"""TRACE-XAI integrated pipeline: one call from task to final XAI report.

    User Task -> Banking Agent -> Execution Trace
        -> 1. Requirement Matching   (Q1, explain/decision)
        -> 2. Deviation Detection    (Q2, explain/deviation)
        -> 3. Trajectory Anomalies   (Q2, explain/anomaly via deviation)
        -> 4. Counterfactual Analysis(Q3, explain/counterfactual)
        -> 5. SLM Rationale          (rationale/, deterministic fallback)
        -> 6. Evidence Validation    (validate/, guardrails)
        -> Final XAI Report (15 sections)

Note on ordering: validation (6) verifies the claims produced by the
rationale layer (5), so it runs after it — the numbered list in the research
brief is the *feature* list; this is the data-flow order.

All components are the validated Phase 1-7 implementations, injected and
composed — nothing is reimplemented here. Each stage is timed so the report
carries its own performance profile.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from .agent.rule_agent import RuleBasedAgent
from .core.trace import ExecutionTrace
from .explain.counterfactual import CounterfactualExplainer
from .explain.decision import DecisionExplainer
from .explain.deviation import DeviationExplainer
from .explain.models import (
    CounterfactualExplanation,
    DecisionExplanation,
    DeviationExplanation,
    RequirementAssessment,
    RequirementStatus,
)
from .banking.results import ExecutionResult
from .rationale.context import RationaleContext, build_context
from .rationale.generator import RationaleGenerator
from .rationale.models import StructuredRationale
from .validate.models import (
    GuardrailDecision,
    OverallValidationStatus,
    ValidationReport,
    VerificationStatus,
)
from .validate.validator import ExplanationValidator


class FinalXAIReport(BaseModel):
    """The integrated report: 15 user-facing sections + embedded sub-reports."""

    # identity (items 1 and 3 also flattened below)
    task_id: str
    trace_id: str
    agent_id: str
    action: str

    # --- the 15 required sections
    task: str = Field(description="1. Task text")
    requirements: list[RequirementAssessment] = Field(
        default_factory=list, description="2. Registered requirements (with status)"
    )
    agent_action: list[str] = Field(
        default_factory=list, description="3. Observed action + outcome"
    )
    decision_rationale: str = Field(default="", description="4. Q1 rationale text")
    requirement_satisfaction: dict[str, str] = Field(
        default_factory=dict, description="5. requirement_id -> status"
    )
    deviations: list[str] = Field(
        default_factory=list, description="6. Q2 deviation summary lines"
    )
    trajectory_anomalies: list[str] = Field(
        default_factory=list, description="7. Detected anomalies"
    )
    counterfactuals: list[str] = Field(
        default_factory=list, description="8. Q3 alternative-action summaries"
    )
    evidence: list[str] = Field(
        default_factory=list, description="9. Evidence index lines"
    )
    validation: ValidationReport = Field(
        description="10. Per-claim verdicts, counts, overall status, guardrail"
    )
    supported_claims: list[str] = Field(
        default_factory=list, description="11. Claim texts with SUPPORTED verdict"
    )
    contradicted_claims: list[str] = Field(
        default_factory=list, description="12. Claim texts with CONTRADICTED verdict"
    )
    unknown_claims: list[str] = Field(
        default_factory=list, description="13. Claim texts with UNKNOWN verdict"
    )
    uncertainty: list[str] = Field(
        default_factory=list, description="14. Confidence/uncertainty lines"
    )
    limitations: list[str] = Field(
        default_factory=list, description="15. Explicit limitations of this report"
    )

    # --- embedded sub-reports (audit trail; nothing is flattened away)
    decision: DecisionExplanation
    deviation: DeviationExplanation
    counterfactual_explanation: CounterfactualExplanation
    rationale: StructuredRationale

    # --- pipeline metadata
    timings_ms: dict[str, float] = Field(default_factory=dict)
    llm_used: bool = Field(
        default=False, description="True if any stage consulted a model"
    )
    method: str = "trace_xai_pipeline_v1"

    # ---------------------------------------------------------- rendering
    def to_text(self) -> str:
        """Human-readable final report with the 15 numbered sections."""
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append(f"TRACE-XAI FINAL REPORT  {self.task_id} / {self.trace_id}")
        lines.append("=" * 72)

        def block(number: int, title: str, body: list[str] | str) -> None:
            lines.append(f"\n[{number}] {title}")
            items = [body] if isinstance(body, str) else body
            lines.extend(f"    {item}" for item in items or ["(none)"])

        block(1, "Task", self.task)
        block(2, "Requirements",
              [f"{a.requirement_id}: {a.requirement_text}" for a in self.requirements])
        block(3, "Agent Action", self.agent_action)
        block(4, "Decision Rationale", self.decision_rationale)
        block(5, "Requirement Satisfaction",
              [f"{rid}: {status}" for rid, status in self.requirement_satisfaction.items()])
        block(6, "Deviations", self.deviations)
        block(7, "Trajectory Anomalies", self.trajectory_anomalies)
        block(8, "Counterfactual Alternatives", self.counterfactuals)
        block(9, "Evidence", self.evidence)
        v = self.validation
        block(10, "Validation Results", [
            f"overall: {v.overall_validation_status.value}",
            f"guardrail: {v.guardrail_decision.value}",
            f"supported/contradicted/unknown: {v.supported_claim_count}/"
            f"{v.contradicted_claim_count}/{v.unknown_claim_count}",
            f"unsupported claim rate: {v.unsupported_claim_rate:.2f}",
        ])
        block(11, "Supported Claims", self.supported_claims)
        block(12, "Contradicted Claims", self.contradicted_claims)
        block(13, "Unknown Claims", self.unknown_claims)
        block(14, "Confidence / Uncertainty", self.uncertainty)
        block(15, "Limitations", self.limitations)
        lines.append("\n" + "-" * 72)
        lines.append("synthetic data only | no hidden chain-of-thought | "
                     "claims are citation-grounded")
        return "\n".join(lines)


class TraceXAIPipeline:
    """Composes the validated Q1-Q4 components behind one call."""

    name = "trace-xai-pipeline-v1"

    def __init__(
        self,
        agent: RuleBasedAgent | None = None,
        requirements=None,
        rationale_provider=None,
        judge_provider=None,
        replay_enabled: bool = True,
    ) -> None:
        # one shared decision explainer: deviation + counterfactual + report
        # all reuse the same Phase 3 computation (no duplication)
        self._agent = agent
        self._decision = DecisionExplainer(requirements=requirements)
        self._deviation = DeviationExplainer(decision_explainer=self._decision)
        self._counterfactual = CounterfactualExplainer(
            decision_explainer=self._decision, requirements=requirements,
            replay_enabled=replay_enabled,
        )
        self._rationale = RationaleGenerator(provider=rationale_provider)
        self._validator = ExplanationValidator(judge_provider=judge_provider)
        self._traces: dict[str, ExecutionTrace] = {}
        self._reports: dict[str, FinalXAIReport] = {}

    # ------------------------------------------------------------- public
    def run(self, task: str) -> tuple[ExecutionResult, FinalXAIReport]:
        """Agent executes the task; the pipeline analyzes the resulting trace."""
        if self._agent is None:
            raise RuntimeError("no agent configured; pass a trace to "
                               "analyze_execution() instead")
        result, trace = self._agent.run_traced(task)
        return result, self.analyze_execution(trace=trace)

    def analyze_execution(
        self, task_id: str | None = None, trace: ExecutionTrace | None = None
    ) -> FinalXAIReport:
        """Single-call analysis: task_id of a previously run task, or a trace."""
        if trace is None:
            if task_id is None:
                raise ValueError("provide task_id or trace")
            trace = self._traces.get(task_id)
            if trace is None:
                known = ", ".join(sorted(self._traces)) or "none"
                raise KeyError(f"unknown task_id {task_id!r}; known: {known}")
        timings: dict[str, float] = {}

        def stage(name: str):
            class _Timer:
                def __enter__(self):
                    self.t0 = time.perf_counter()
                    return self
                def __exit__(self, *exc):
                    timings[name] = round((time.perf_counter() - self.t0) * 1000, 3)
            return _Timer()

        with stage("decision"):
            decision = self._decision.explain(trace)
        with stage("deviation"):
            deviation = self._deviation.explain(trace)
        with stage("counterfactual"):
            counterfactual = self._counterfactual.explain(trace)
        with stage("rationale"):
            context = build_context(trace, decision, deviation, counterfactual)
            rationale = self._rationale.generate(
                trace, decision, deviation, counterfactual, context=context)
        with stage("validation"):
            validation = self._validator.validate(rationale, context)

        report = self._assemble(trace, context, decision, deviation,
                                counterfactual, rationale, validation, timings)
        self._traces[trace.task_id] = trace
        self._reports[trace.task_id] = report
        return report

    def get_report(self, task_id: str) -> FinalXAIReport:
        if task_id not in self._reports:
            raise KeyError(f"no report for {task_id!r}")
        return self._reports[task_id]

    # ---------------------------------------------------------- internals
    @staticmethod
    def _assemble(trace, context: RationaleContext, decision, deviation,
                  counterfactual, rationale: StructuredRationale,
                  validation: ValidationReport,
                  timings: dict[str, float]) -> FinalXAIReport:
        by_status: dict[VerificationStatus, list[str]] = {
            VerificationStatus.SUPPORTED: [],
            VerificationStatus.CONTRADICTED: [],
            VerificationStatus.UNKNOWN: [],
        }
        for verdict in validation.verifications:
            by_status[verdict.verification_status].append(verdict.claim)

        limitations = _limitations(
            decision, deviation, counterfactual, rationale, validation)

        return FinalXAIReport(
            task_id=trace.task_id, trace_id=trace.trace_id,
            agent_id=trace.agent_id, action=rationale.action,
            task=trace.task_text,
            requirements=decision.requirements,
            agent_action=context.agent_action,
            decision_rationale=rationale.decision_rationale,
            requirement_satisfaction={
                a.requirement_id: a.status.value for a in decision.requirements
            },
            deviations=context.deviations,
            trajectory_anomalies=context.trajectory_anomalies,
            counterfactuals=context.counterfactuals,
            evidence=[e.ref_line() for e in context.evidence],
            validation=validation,
            supported_claims=by_status[VerificationStatus.SUPPORTED],
            contradicted_claims=by_status[VerificationStatus.CONTRADICTED],
            unknown_claims=by_status[VerificationStatus.UNKNOWN],
            uncertainty=context.uncertainty,
            limitations=limitations,
            decision=decision, deviation=deviation,
            counterfactual_explanation=counterfactual, rationale=rationale,
            timings_ms=timings,
            llm_used=rationale.llm_used or validation.judge_used,
        )


def _limitations(decision, deviation, counterfactual, rationale,
                 validation) -> list[str]:
    """Deterministic, explicit limitations for this specific report."""
    out: list[str] = []
    unknown_reqs = [a.requirement_id for a in decision.requirements
                    if a.status is RequirementStatus.UNKNOWN]
    if unknown_reqs:
        out.append(f"requirement(s) {', '.join(unknown_reqs)} could not be "
                   f"evaluated (missing evidence)")
    if deviation.relation.value in ("unknown", "possible"):
        out.append(f"anomaly-violation association is '{deviation.relation.value}' "
                   f"— no causal relationship is claimed")
    if not counterfactual.counterfactuals:
        out.append("no counterfactual alternatives were generated for this trace")
    elif any(cf.replay is None or not cf.replay.performed
             for cf in counterfactual.counterfactuals):
        out.append("some counterfactuals are predictions only (replay not "
                   "performed); simulated != executed")
    if validation.unknown_claim_count:
        out.append(f"{validation.unknown_claim_count} claim(s) are unverifiable "
                   f"from the recorded evidence (UNKNOWN)")
    if validation.contradicted_claim_count:
        out.append(f"{validation.contradicted_claim_count} claim(s) are "
                   f"CONTRADICTED by evidence — guardrail "
                   f"{validation.guardrail_decision.value}")
    if rationale.llm_used:
        out.append("rationale text was generated by a local SLM; phrasing may "
                   "vary (verdicts are citation-grounded)")
    if validation.judge_used:
        out.append("the LLM judge contributed verdicts (advisory, confidence "
                   "capped; deterministic rules take precedence)")
    out.append("all data is synthetic; controlled replays do not reconstruct "
               "full ledger history")
    return out


def analyze_execution(task: str, agent: RuleBasedAgent | None = None,
                      **pipeline_kwargs) -> FinalXAIReport:
    """One-call convenience: run a task and return its final XAI report."""
    pipeline = TraceXAIPipeline(agent=agent or RuleBasedAgent(), **pipeline_kwargs)
    _, report = pipeline.run(task)
    return report
