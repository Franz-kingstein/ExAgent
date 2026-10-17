"""Explain package: evidence-grounded explainability (Q1 decision first)."""

from .decision import DecisionExplainer, explain_decision
from .matcher import RequirementMatcher, TransferFacts, extract_facts
from .models import (
    DecisionExplanation,
    EvidenceItem,
    RequirementAssessment,
    RequirementStatus,
)

__all__ = [
    "DecisionExplainer",
    "explain_decision",
    "RequirementMatcher",
    "TransferFacts",
    "extract_facts",
    "DecisionExplanation",
    "EvidenceItem",
    "RequirementAssessment",
    "RequirementStatus",
]
