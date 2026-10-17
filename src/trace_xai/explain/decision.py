"""Decision explainability (Q1): requirement-to-action matching.

Deterministic and auditable: given a trace and the requirement registry, it
assesses every applicable requirement against observable evidence and composes
a concise rationale. No LLM is involved (llm_used=False by design).
"""

from __future__ import annotations

from ..core.trace import ExecutionTrace
from ..requirements.model import RequirementSet, load_requirements
from .matcher import RequirementMatcher, extract_facts
from .models import DecisionExplanation, RequirementAssessment

_NO_APPLICABLE = (
    "No registered requirements apply to the observed action '{action}'."
)


class DecisionExplainer:
    """Builds evidence-grounded decision explanations from traces."""

    name = "requirement-action-matcher"

    def __init__(self, requirements: RequirementSet | None = None) -> None:
        self._requirements = requirements or load_requirements()
        self._matcher = RequirementMatcher()

    @property
    def requirements(self) -> RequirementSet:
        return self._requirements

    def explain(self, trace: ExecutionTrace) -> DecisionExplanation:
        facts = extract_facts(trace)
        action = facts.action or "UNKNOWN"
        applicable = self._requirements.applicable_to(facts.action)

        assessments = [
            self._matcher.evaluate(trace, facts, requirement)
            for requirement in applicable
        ]
        rationale = (
            _compose_rationale(assessments)
            if assessments
            else _NO_APPLICABLE.format(action=action)
        )
        return DecisionExplanation(
            task_id=trace.task_id,
            trace_id=trace.trace_id,
            task_text=trace.task_text,
            agent_id=trace.agent_id,
            action=action,
            requirements=assessments,
            rationale=rationale,
            method="requirement_to_action_matching_v1",
            llm_used=False,
        )


def _compose_rationale(assessments: list[RequirementAssessment]) -> str:
    """Join per-requirement clauses into one concise rationale sentence."""
    clauses = []
    for assessment in assessments:
        clause = assessment.explanation.split(" Evidence:")[0]
        if assessment.status.value == "UNKNOWN" and assessment.conflicting_evidence:
            clause = f"evidence conflicted about {assessment.requirement_id}"
        elif assessment.status.value == "UNKNOWN":
            clause = f"there was no observable evidence about {assessment.requirement_id}"
        clauses.append(clause.rstrip("."))
    rationale = "; ".join(clauses) + "."
    return rationale[0].upper() + rationale[1:]


def explain_decision(
    trace: ExecutionTrace, requirements: RequirementSet | None = None
) -> DecisionExplanation:
    """Convenience one-call API."""
    return DecisionExplainer(requirements=requirements).explain(trace)
