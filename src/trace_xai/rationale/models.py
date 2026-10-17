"""Output models for the SLM rationale layer (user-facing, evidence-grounded).

The rationale is a *user-facing summary* of already-computed explanations
(Q1 decision, Q2 deviation, Q3 counterfactual). It is never a chain-of-thought
and never claims access to hidden reasoning: every claim is a label
(FACT/EVIDENCE/INFERENCE/UNKNOWN) with citations that must resolve to the
evidence supplied to the model.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClaimLabel(str, Enum):
    """Explicit epistemic label required by the TRACE-XAI core principles."""

    FACT = "FACT"              # verbatim from a supplied record
    EVIDENCE = "EVIDENCE"      # supported by a cited observation
    INFERENCE = "INFERENCE"    # derived interpretation (still cited)
    UNKNOWN = "UNKNOWN"        # explicitly missing information


class RationaleClaim(BaseModel):
    """One factual claim; must cite supplied evidence unless labelled UNKNOWN."""

    text: str
    label: ClaimLabel
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="E-numbers from the evidence index or REQ-/ANOM-/CF- ids",
    )


class StructuredRationale(BaseModel):
    """The five required user-facing sections + auditable claims.

    `grounded=False` means at least one citation did not resolve to the
    supplied evidence (the model invented it); such claims are quarantined in
    `unsupported_claims` and the rationale must not be trusted until reviewed.
    """

    task_id: str
    trace_id: str
    agent_id: str
    action: str

    decision_rationale: str
    deviation_explanation: str
    counterfactual_summary: str
    evidence_summary: str
    uncertainty_statement: str

    claims: list[RationaleClaim] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims/refs that failed citation resolution (quarantined)",
    )
    grounded: bool = True

    provider: str = Field(description="Model provider that produced the text")
    llm_used: bool = False
    provenance: dict = Field(
        default_factory=dict, description="synthetic flag, methods, source layers"
    )
    notes: list[str] = Field(
        default_factory=list, description="Fallback/parse/grounding events"
    )
    method: str = "slm_structured_rationale_v1"
