"""Validate package: Q4 explanation validation.

Pipeline: claim extraction -> evidence retrieval -> claim verification
(deterministic rules, lexical NLI backstop, optional LLM judge) ->
SUPPORTED/CONTRADICTED/UNKNOWN -> guardrail decision (PASS/FLAG/BLOCK).
The validator identifies unsupported claims; it never rewrites them.
"""

from .extract import ExtractedClaim, extract_claims
from .judge import LLMJudge
from .models import (
    ClaimVerification,
    GuardrailDecision,
    OverallValidationStatus,
    ValidationReport,
    VerificationMethod,
    VerificationStatus,
)
from .validator import ExplanationValidator, validate_explanation

__all__ = [
    "extract_claims",
    "ExtractedClaim",
    "LLMJudge",
    "ClaimVerification",
    "GuardrailDecision",
    "OverallValidationStatus",
    "ValidationReport",
    "VerificationMethod",
    "VerificationStatus",
    "ExplanationValidator",
    "validate_explanation",
]
