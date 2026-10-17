"""Explain package: evidence-grounded explainability (Q1 decision, Q2 deviation)."""

from .decision import DecisionExplainer, explain_decision
from .deviation import DeviationExplainer, explain_deviation
from .anomaly import TrajectoryAnomalyDetector
from .matcher import RequirementMatcher, TransferFacts, extract_facts
from .models import (
    Anomaly,
    AnomalyType,
    DecisionExplanation,
    DeviationCategory,
    DeviationExplanation,
    EvidenceItem,
    RelationLabel,
    RequirementAssessment,
    RequirementStatus,
    Severity,
)

__all__ = [
    "DecisionExplainer",
    "explain_decision",
    "DeviationExplainer",
    "explain_deviation",
    "TrajectoryAnomalyDetector",
    "RequirementMatcher",
    "TransferFacts",
    "extract_facts",
    "Anomaly",
    "AnomalyType",
    "DecisionExplanation",
    "DeviationCategory",
    "DeviationExplanation",
    "EvidenceItem",
    "RelationLabel",
    "RequirementAssessment",
    "RequirementStatus",
    "Severity",
]
