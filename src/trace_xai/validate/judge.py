"""LLM/SLM-as-judge: semantic verification of claims the rules cannot decide.

Used ONLY when deterministic rules and the lexical backstop return UNKNOWN,
and only when a judge provider is configured. Constraints:

- The judge receives the claim and the *verbatim cited evidence lines* only —
  no outside knowledge is invited; it is told to answer UNKNOWN when the
  evidence does not settle the claim.
- Its verdict is never accepted over a deterministic verdict (deterministic
  rules run first and win).
- A SUPPORTED verdict with zero cited evidence is forced back to UNKNOWN
  (guardrail: evidence-free support is impossible).
- Confidence is capped at JUDGE_CONFIDENCE_CAP so a single model call cannot
  masquerade as ground truth.
"""

from __future__ import annotations

import json
import re

from ..rationale.provider import ProviderError, RationaleProvider
from .models import VerificationStatus

JUDGE_CONFIDENCE_CAP = 0.75

JUDGE_SYSTEM = """\
You are a strict verification judge for an explainability system. You receive \
ONE claim and the verbatim evidence lines that were cited for it.

Decide:
- SUPPORTED   — the evidence explicitly entails the claim
- CONTRADICTED — the evidence explicitly conflicts with the claim
- UNKNOWN     — the evidence is missing, insufficient, or does not settle it

Rules: use ONLY the provided evidence; never use outside knowledge; when in \
doubt answer UNKNOWN; a claim about a fact absent from the evidence is NEVER \
SUPPORTED. Output STRICT JSON only:
{"status": "SUPPORTED"|"CONTRADICTED"|"UNKNOWN", "confidence": 0.0-1.0,
 "explanation": "one sentence citing the decisive evidence"}
"""


class LLMJudge:
    name = "llm-judge-v1"

    def __init__(self, provider: RationaleProvider) -> None:
        self._provider = provider

    def judge(self, claim: str, evidence_lines: list[str]) -> tuple[
            VerificationStatus, float, str] | None:
        """Return (status, confidence, explanation) or None if unusable."""
        if not evidence_lines:
            # guardrail: support without evidence is impossible
            return (VerificationStatus.UNKNOWN, 0.1,
                    "judge skipped: claim has no cited evidence to judge against")
        user = ("CLAIM:\n" + claim + "\n\nCITED EVIDENCE:\n"
                + "\n".join(f"- {line}" for line in evidence_lines))
        try:
            raw = self._provider.generate(JUDGE_SYSTEM, user)
        except ProviderError:
            return None  # judge unavailable -> claim stays UNKNOWN
        data = self._parse(raw)
        if data is None:
            return None
        status_raw = str(data.get("status", "")).upper()
        try:
            status = VerificationStatus(status_raw)
        except ValueError:
            status = VerificationStatus.UNKNOWN
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(JUDGE_CONFIDENCE_CAP, confidence))
        explanation = str(data.get("explanation") or "judge gave no explanation")
        return status, confidence, f"llm judge: {explanation}"

    @staticmethod
    def _parse(raw: str) -> dict | None:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None
