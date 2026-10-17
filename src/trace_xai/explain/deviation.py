"""Q2 behavior-deviation explainability: hybrid of requirement checking (Phase 3)
and trajectory anomaly detection (this phase).

Answers "what went wrong during the agent's execution?" with one of:
A. REQUIREMENT_VIOLATION   — explicit requirement violated
B. TRAJECTORY_ANOMALY      — abnormal/unexpected trajectory behavior
C. BOTH
D. INSUFFICIENT_EVIDENCE   — cannot be determined from available evidence
   (plus NONE when everything is clean)

Association between anomalies and violations is labelled
confirmed / possible / unknown — never a causal claim. The trace only
establishes that events co-occur at the same tool call (confirmed association)
or not (possible); causation is not asserted.
"""

from __future__ import annotations

from ..core.trace import ExecutionTrace
from .anomaly import TrajectoryAnomalyDetector
from .decision import DecisionExplainer
from .models import (
    Anomaly,
    DecisionExplanation,
    DeviationCategory,
    DeviationExplanation,
    RelationLabel,
    RequirementAssessment,
    RequirementStatus,
)


class DeviationExplainer:
    """Hybrid Q2 explainer (deterministic; no LLM)."""

    name = "hybrid-deviation-explainer-v1"

    def __init__(
        self,
        decision_explainer: DecisionExplainer | None = None,
        anomaly_detector: TrajectoryAnomalyDetector | None = None,
    ) -> None:
        self._decision = decision_explainer or DecisionExplainer()
        self._detector = anomaly_detector or TrajectoryAnomalyDetector()

    def explain(self, trace: ExecutionTrace) -> DeviationExplanation:
        decision = self._decision.explain(trace)
        anomalies = self._detector.detect(trace)
        return self._compose(trace, decision, anomalies)

    # ------------------------------------------------------------ synthesis
    def _compose(
        self,
        trace: ExecutionTrace,
        decision: DecisionExplanation,
        anomalies: list[Anomaly],
    ) -> DeviationExplanation:
        violations = [
            a for a in decision.requirements if a.status is RequirementStatus.VIOLATED
        ]
        unknowns = [
            a for a in decision.requirements if a.status is RequirementStatus.UNKNOWN
        ]
        category = self._categorize(violations, unknowns, anomalies)
        relation = self._relate(violations, unknowns, anomalies)
        combined = self._combine(violations, unknowns, anomalies, relation, category)

        return DeviationExplanation(
            task_id=trace.task_id,
            trace_id=trace.trace_id,
            task_text=trace.task_text,
            agent_id=trace.agent_id,
            action=decision.action,
            category=category,
            requirement_violations=violations,
            unknown_requirement_ids=[u.requirement_id for u in unknowns],
            anomalies=anomalies,
            relation=relation,
            combined_explanation=combined,
            method="hybrid_requirement_and_trajectory_analysis_v1",
            llm_used=False,
        )

    @staticmethod
    def _categorize(violations, unknowns, anomalies) -> DeviationCategory:
        if violations and anomalies:
            return DeviationCategory.BOTH
        if violations:
            return DeviationCategory.REQUIREMENT_VIOLATION
        if anomalies:
            return DeviationCategory.TRAJECTORY_ANOMALY
        if unknowns:
            return DeviationCategory.INSUFFICIENT_EVIDENCE
        return DeviationCategory.NONE

    @staticmethod
    def _relate(violations, unknowns, anomalies) -> RelationLabel:
        """Association only — the trace establishes co-location, not causation."""
        if not anomalies:
            return RelationLabel.NONE
        if not violations:
            return RelationLabel.UNKNOWN if unknowns else RelationLabel.NONE
        violation_events = {
            e.event_id for v in violations for e in v.supporting_evidence
        }
        for anomaly in anomalies:
            if set(anomaly.event_ids) & violation_events:
                return RelationLabel.CONFIRMED  # same tool call, association only
        return RelationLabel.POSSIBLE

    @staticmethod
    def _combine(violations, unknowns, anomalies, relation, category) -> str:
        if category is DeviationCategory.NONE:
            return ("No deviation detected: all applicable requirements were "
                    "satisfied and no trajectory anomalies were found.")

        parts: list[str] = []
        if violations:
            for v in violations:
                clause = v.explanation.split(" Evidence:")[0].rstrip()
                parts.append(f"Requirement {v.requirement_id} was VIOLATED: {clause}")
        elif unknowns:
            ids = ", ".join(u.requirement_id for u in unknowns)
            parts.append(
                f"No requirement violation could be established; evidence was "
                f"insufficient for {ids}."
            )
        else:
            parts.append("No requirement violation was detected.")

        if anomalies:
            count = len(anomalies)
            noun = "anomaly" if count == 1 else "anomalies"
            parts.append(f"The execution trace also contains {count} {noun}:")
            for a in anomalies:
                parts.append(
                    f"  - {a.anomaly_id} [{a.anomaly_type.value}, "
                    f"severity {a.severity.value}, confidence {a.confidence:.2f}] "
                    f"{a.explanation}"
                )

        if relation is RelationLabel.CONFIRMED:
            parts.append(
                "Association: an anomaly and a violation involve the same tool "
                "call (confirmed). No causal relationship is claimed."
            )
        elif relation is RelationLabel.POSSIBLE:
            parts.append(
                "Association: the anomaly and the violation occur at different "
                "points in the trace (possible, not established)."
            )
        elif relation is RelationLabel.UNKNOWN:
            parts.append(
                "Association: whether the anomaly relates to a requirement "
                "violation is unknown due to insufficient evidence."
            )
        return "\n".join(parts)


def explain_deviation(trace: ExecutionTrace) -> DeviationExplanation:
    """Convenience one-call API."""
    return DeviationExplainer().explain(trace)
