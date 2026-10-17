"""Explain package: evidence-grounded explainability (Q1-Q3)."""

from .anomaly import TrajectoryAnomalyDetector
from .counterfactual import (
    ControlledReplayEngine,
    CounterfactualExplainer,
    SandboxSimulator,
)
from .decision import DecisionExplainer, explain_decision
from .deviation import DeviationExplainer, explain_deviation
from .matcher import RequirementMatcher, TransferFacts, extract_facts
from .models import (
    ActionSpec,
    Anomaly,
    AnomalyType,
    Counterfactual,
    CounterfactualExplanation,
    CounterfactualImpact,
    DecisionExplanation,
    DeviationCategory,
    DeviationExplanation,
    EvidenceItem,
    PredictedOutcome,
    RelationLabel,
    ReplayRecord,
    RequirementAssessment,
    RequirementComparisonItem,
    RequirementStatus,
    Severity,
)

__all__ = [
    "DecisionExplainer",
    "explain_decision",
    "DeviationExplainer",
    "explain_deviation",
    "CounterfactualExplainer",
    "SandboxSimulator",
    "ControlledReplayEngine",
    "TrajectoryAnomalyDetector",
    "RequirementMatcher",
    "TransferFacts",
    "extract_facts",
    "ActionSpec",
    "Anomaly",
    "AnomalyType",
    "Counterfactual",
    "CounterfactualExplanation",
    "CounterfactualImpact",
    "DecisionExplanation",
    "DeviationCategory",
    "DeviationExplanation",
    "EvidenceItem",
    "PredictedOutcome",
    "RelationLabel",
    "ReplayRecord",
    "RequirementAssessment",
    "RequirementComparisonItem",
    "RequirementStatus",
    "Severity",
]
