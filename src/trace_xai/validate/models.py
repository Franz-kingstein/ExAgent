"""Output models for explanation validation (Q4).

Every extracted claim gets a verification verdict; the report aggregates
counts, an unsupported-claim rate, an overall status, and a guardrail
decision. The validator never rewrites explanations — it judges them.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"        # cited evidence entails the claim
    CONTRADICTED = "CONTRADICTED"  # cited evidence conflicts with the claim
    UNKNOWN = "UNKNOWN"            # evidence insufficient / not resolvable


class VerificationMethod(str, Enum):
    DETERMINISTIC_RULE = "deterministic_rule"  # structured fact comparison
    LEXICAL_NLI = "lexical_nli"                # lightweight textual-entailment backstop
    LLM_JUDGE = "llm_judge"                    # semantic claims only, never trusted blindly
    GUARDRAIL = "guardrail"                    # citation/structure checks


class ClaimVerification(BaseModel):
    """Verdict for one claim (the unit the pipeline verifies)."""

    claim: str
    evidence: list[str] = Field(
        default_factory=list,
        description="Rendered cited evidence lines (REF = field = value)",
    )
    verification_status: VerificationStatus
    verification_method: VerificationMethod
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class OverallValidationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"                    # every verified claim supported
    CONTRADICTED = "CONTRADICTED"              # at least one contradiction
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # unknowns, or nothing to verify


class GuardrailDecision(str, Enum):
    PASS = "PASS"      # explanation may be shown as-is
    FLAG = "FLAG"      # shown only with uncertainty caveats
    BLOCK = "BLOCK"    # must not be shown until corrected


class ValidationReport(BaseModel):
    task_id: str
    trace_id: str

    verifications: list[ClaimVerification] = Field(default_factory=list)
    supported_claim_count: int = 0
    contradicted_claim_count: int = 0
    unknown_claim_count: int = 0
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)
    overall_validation_status: OverallValidationStatus
    guardrail_decision: GuardrailDecision

    judge_used: bool = False
    notes: list[str] = Field(default_factory=list)
    method: str = "claim_verification_pipeline_v1"
