"""Explanation presentation modes (Innovation A).

Four audience-adapted views of the *same* validated explanation:

- ``TECHNICAL``  — developers/researchers: requirement ids, trace/event refs,
                   verification methods, confidences, guardrail.
- ``BUSINESS``   — managers/auditors: policy compliance, business impact,
                   operational risk, reason for the action.
- ``NORMAL``     — everyday language.
- ``ELI5``       — very simple language plus a clearly-labelled analogy.

Design decision (important for the research claims):

The wording layer is a **pure deterministic function of the validated
``FinalXAIReport``**. No model is consulted while re-wording. This is what
*guarantees* the invariant the project needs: wording may adapt to the
audience, but evidence references, verification statuses, counts and
conclusions cannot drift, because they are copied, never recomputed.

Audience-adaptive wording is a *presentation* technique, not a novel
explainability algorithm — see docs/innovation_summary.md.
"""

from __future__ import annotations

from enum import Enum


class PresentationMode(str, Enum):
    """Selectable explanation audience."""

    TECHNICAL = "technical"
    BUSINESS = "business"
    NORMAL = "normal"
    ELI5 = "eli5"


class ClaimKind(str, Enum):
    """Epistemic class of a presented item — never blurred across modes.

    FACT        — settled by recorded evidence (observation or its negation)
    INFERENCE   — derived/predicted (counterfactual outcome, association label)
    UNKNOWN     — not determinable from the recorded evidence
    """

    FACT = "FACT"
    INFERENCE = "INFERENCE"
    UNKNOWN = "UNKNOWN"


MODE_AUDIENCE: dict[PresentationMode, str] = {
    PresentationMode.TECHNICAL: "developers and AI researchers",
    PresentationMode.BUSINESS: "managers, auditors and decision-makers",
    PresentationMode.NORMAL: "general audience (everyday language)",
    PresentationMode.ELI5: "non-technical audience (very simple language)",
}

MODE_DESCRIPTION: dict[PresentationMode, str] = {
    PresentationMode.TECHNICAL:
        "Requirement ids, trace events, violations, counterfactual findings, "
        "evidence references and verification status.",
    PresentationMode.BUSINESS:
        "Policy compliance, business impact, operational risk and the reason "
        "for the agent's action.",
    PresentationMode.NORMAL:
        "Clear, everyday language describing what happened and why.",
    PresentationMode.ELI5:
        "Very simple language with an analogy where useful.",
}


# --------------------------------------------------------------- wording
# Plain-language status verbalisation. Only *status words* are re-worded;
# the underlying recorded facts (requirement text, evidence, numbers) are
# always carried through unchanged from the validated report.

_STATUS_TECHNICAL = {
    "SATISFIED": "SATISFIED", "VIOLATED": "VIOLATED", "UNKNOWN": "UNKNOWN",
}
_STATUS_BUSINESS = {
    "SATISFIED": "compliant",
    "VIOLATED": "not compliant (policy breach)",
    "UNKNOWN": "not assessable from recorded evidence",
}
_STATUS_NORMAL = {
    "SATISFIED": "the rule was followed",
    "VIOLATED": "the rule was not followed",
    "UNKNOWN": "we do not have enough recorded information to say",
}
_STATUS_ELI5 = {
    "SATISFIED": "the rule was followed",
    "VIOLATED": "the rule was broken",
    "UNKNOWN": "nobody knows from what we wrote down",
}

_STATUS_BY_MODE: dict[PresentationMode, dict[str, str]] = {
    PresentationMode.TECHNICAL: _STATUS_TECHNICAL,
    PresentationMode.BUSINESS: _STATUS_BUSINESS,
    PresentationMode.NORMAL: _STATUS_NORMAL,
    PresentationMode.ELI5: _STATUS_ELI5,
}


def status_words(mode: PresentationMode, status: str) -> str:
    """Audience-appropriate wording for a *recorded* requirement status.

    Only the status word changes between modes. ``status`` itself is always
    the recorded value (SATISFIED/VIOLATED/UNKNOWN) — presentation never
    alters it, so an UNKNOWN can never be presented as satisfied.
    """
    table = _STATUS_BY_MODE[mode]
    return table.get(status.upper(), status)


# Rendering verbs used to introduce a block, per mode.
INTRO: dict[PresentationMode, dict[str, str]] = {
    PresentationMode.TECHNICAL: {
        "action": "observed action",
        "rationale": "decision rationale (evidence-grounded)",
        "requirements": "requirement matching",
        "deviations": "deviation detection",
        "anomalies": "trajectory anomalies",
        "counterfactuals": "counterfactual analysis",
        "claims": "claim verification",
        "evidence": "evidence index",
        "uncertainty": "uncertainty",
        "limitations": "limitations",
    },
    PresentationMode.BUSINESS: {
        "action": "What the agent did",
        "rationale": "Why the agent acted this way",
        "requirements": "Policy compliance",
        "deviations": "Operational findings",
        "anomalies": "Process irregularities",
        "counterfactuals": "Alternative actions considered",
        "claims": "How well the explanation is evidenced",
        "evidence": "Supporting records",
        "uncertainty": "Open questions",
        "limitations": "What this report does not show",
    },
    PresentationMode.NORMAL: {
        "action": "What happened",
        "rationale": "Why it happened",
        "requirements": "Rules involved",
        "deviations": "Anything unusual",
        "anomalies": "Strange patterns in the steps",
        "counterfactuals": "What else could have happened",
        "claims": "Is the explanation backed by the records?",
        "evidence": "The records used",
        "uncertainty": "What we are unsure about",
        "limitations": "Things to keep in mind",
    },
    PresentationMode.ELI5: {
        "action": "What the robot did",
        "rationale": "Why the robot did it",
        "requirements": "The rules it had to follow",
        "deviations": "Anything that looked wrong",
        "anomalies": "Repeated or odd steps",
        "counterfactuals": "Other things it could have done",
        "claims": "Did we check the robot's story?",
        "evidence": "The notes we checked",
        "uncertainty": "What we are not sure about",
        "limitations": "What this picture cannot tell you",
    },
}
