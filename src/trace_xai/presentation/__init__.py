"""Audience-adapted explanation presentation (Innovation A).

Four modes — TECHNICAL, BUSINESS, NORMAL, ELI5 — generated from the *same*
validated ``FinalXAIReport``. Wording adapts to the audience; evidence
references, verification statuses, counts and conclusions are copied, never
recomputed, so simplification can never turn an UNKNOWN into a SUPPORTED.

Usage::

    from trace_xai import analyze_execution
    report = analyze_execution("Transfer ₹10,000 to beneficiary BOB-001.")
    presented = report.present("business")     # or .presented_all()
    print(presented.to_text())
"""

from .modes import (
    MODE_AUDIENCE,
    MODE_DESCRIPTION,
    ClaimKind,
    PresentationMode,
    status_words,
)
from .presenter import (
    PresentedBlock,
    PresentedClaim,
    PresentedExplanation,
    PresentedLine,
    PresentedRequirement,
    present_all,
    present_report,
)

__all__ = [
    "PresentationMode",
    "ClaimKind",
    "MODE_AUDIENCE",
    "MODE_DESCRIPTION",
    "status_words",
    "PresentedLine",
    "PresentedBlock",
    "PresentedRequirement",
    "PresentedClaim",
    "PresentedExplanation",
    "present_report",
    "present_all",
]
