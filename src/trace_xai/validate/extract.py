"""Claim extraction from generated explanations.

Primary source: the labelled ``claims`` array of a Phase 6
``StructuredRationale`` (already citation-annotated). Secondary source:
sentences of the five section texts that contain citable ids — prose claims
are validated too, not just the structured array.

UNKNOWN-labelled claims are uncertainty *statements*, not factual
assertions; they are skipped by the validator (recorded in notes), because
there is nothing to verify — a claim that "we do not know" cannot itself be
supported or contradicted by missing evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..rationale.models import ClaimLabel, StructuredRationale

_CITATION_PATTERN = re.compile(r"\b(?:E\d+|REQ-[A-Z]+-\d+|ANOM-\d{4}|CF-\d{4})\b")


@dataclass
class ExtractedClaim:
    text: str
    refs: list[str] = field(default_factory=list)
    source: str = "claims"  # "claims" (structured array) | "section:<name>"
    label: str | None = None  # original label when it came from the array


def extract_claims(rationale: StructuredRationale) -> list[ExtractedClaim]:
    """Claims from the array + citable sentences from the five sections."""
    out: list[ExtractedClaim] = []
    seen: set[str] = set()

    for claim in rationale.claims:
        text = claim.text.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(ExtractedClaim(
                text=text, refs=list(claim.evidence_refs),
                source="claims", label=claim.label.value,
            ))

    sections = {
        "decision_rationale": rationale.decision_rationale,
        "deviation_explanation": rationale.deviation_explanation,
        "counterfactual_summary": rationale.counterfactual_summary,
        "evidence_summary": rationale.evidence_summary,
        "uncertainty_statement": rationale.uncertainty_statement,
    }
    for name, text in sections.items():
        for sentence in _split_sentences(text):
            if sentence in seen:
                continue
            refs = sorted(set(_CITATION_PATTERN.findall(sentence)))
            if not refs:
                continue  # uncitable prose is handled by guardrails, not here
            seen.add(sentence)
            out.append(ExtractedClaim(text=sentence, refs=refs, source=f"section:{name}"))
    return out


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]
