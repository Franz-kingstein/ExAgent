"""Rationale generator: SLM over explicit evidence, with a deterministic floor.

Pipeline (per execution trace):

1. ``build_context``  — deterministic evidence packet from trace + Q1/Q2/Q3.
2. ``render_prompt``  — structured system/user prompts (no CoT requests).
3. provider call      — optional; any ``RationaleProvider``; failures are
                        contained (ProviderError -> fallback).
4. parse + ground     — strict JSON parse; every claim citation must resolve
                        to [ALLOWED CITATION REFS]; ungrounded claims are
                        quarantined (never presented as supported).
5. deterministic fallback — if no provider, or the model output is unusable
                        (parse failure, empty sections), the five sections are
                        composed directly from the evidence packet
                        (``llm_used=False``).

Honesty invariant: this layer *communicates* what Q1/Q2/Q3 established; it
never adds facts. ``grounded=False`` + ``unsupported_claims`` make any model
invention visible instead of silently trusted.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..core.trace import ExecutionTrace
from ..explain.models import (
    CounterfactualExplanation,
    DecisionExplanation,
    DeviationExplanation,
)
from .context import RationaleContext, build_context
from .models import ClaimLabel, RationaleClaim, StructuredRationale
from .provider import ProviderError, RationaleProvider
from .template import OUTPUT_KEYS, render_prompt

_REF_PATTERN = re.compile(r"^(?:E\d+|REQ-[A-Z]+-\d+|ANOM-\d{4}|CF-\d{4})$")


class RationaleGenerator:
    """Generates the user-facing rationale (optionally via a replaceable SLM)."""

    name = "rationale-generator-v1"

    def __init__(self, provider: RationaleProvider | None = None) -> None:
        self._provider = provider

    # ------------------------------------------------------------- public
    def generate(
        self,
        trace: ExecutionTrace,
        decision: DecisionExplanation | None = None,
        deviation: DeviationExplanation | None = None,
        counterfactual: CounterfactualExplanation | None = None,
        context: RationaleContext | None = None,
    ) -> StructuredRationale:
        ctx = context or build_context(trace, decision, deviation, counterfactual)

        if self._provider is None:
            return self._fallback(ctx, notes=["no provider configured; "
                                              "deterministic composition used"])

        system_prompt, user_prompt = render_prompt(ctx)
        try:
            raw = self._provider.generate(system_prompt, user_prompt)
        except ProviderError as exc:
            return self._fallback(ctx, notes=[f"provider failed: {exc}; "
                                              "deterministic composition used"])

        parsed = self._parse(raw)
        if parsed is None:
            return self._fallback(ctx, notes=["model output was not valid strict "
                                              "JSON; deterministic composition used"])

        claims, unsupported = self._ground(parsed.get("claims") or [], ctx)
        sections = self._sections(parsed, ctx)
        # prose-level hallucination net: ids cited inside the five sections
        # must also resolve to the supplied packet
        unsupported.extend(self._scan_sections(sections, ctx))
        notes: list[str] = []
        if not parsed.get("claims"):
            notes.append("model returned no claims array; section-level "
                         "citation check applied instead")
        if unsupported:
            notes.append(f"{len(unsupported)} unresolvable citation(s) detected "
                         f"(claims quarantined; sections flagged via grounded=False)")
        grounded = not unsupported
        return StructuredRationale(
            task_id=ctx.task_id, trace_id=ctx.trace_id, agent_id=ctx.agent_id,
            action=self._action_label(ctx),
            decision_rationale=sections["decision_rationale"],
            deviation_explanation=sections["deviation_explanation"],
            counterfactual_summary=sections["counterfactual_summary"],
            evidence_summary=sections["evidence_summary"],
            uncertainty_statement=sections["uncertainty_statement"],
            claims=claims,
            unsupported_claims=unsupported,
            grounded=grounded,
            provider=self._provider.name,
            llm_used=True,
            provenance=self._provenance(ctx),
            notes=notes,
        )

    # ------------------------------------------------------------ internals
    @staticmethod
    def _parse(raw: str) -> dict[str, Any] | None:
        """Strict JSON parse; tolerates ```json fences, rejects anything else."""
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # last resort: first {...} block
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _scan_sections(sections: dict[str, str], ctx: RationaleContext) -> list[str]:
        """Find ids cited in prose that do not resolve to the packet."""
        allowed = ctx.allowed_refs()
        issues: list[str] = []
        token_pattern = re.compile(r"\b(?:E\d+|REQ-[A-Z]+-\d+|ANOM-\d{4}|CF-\d{4})\b")
        for key, text in sections.items():
            for token in sorted(set(token_pattern.findall(text))):
                if token not in allowed:
                    issues.append(f"{key}: section cites unknown id {token}")
        return issues

    @staticmethod
    def _sections(parsed: dict[str, Any], ctx: RationaleContext) -> dict[str, str]:
        """Fetch the five sections; empty/missing -> deterministic section."""
        fallback = RationaleGenerator._deterministic_sections(ctx)
        out: dict[str, str] = {}
        for key in OUTPUT_KEYS:
            if key == "claims":
                continue
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
            else:
                out[key] = fallback[key]
        return out

    @staticmethod
    def _ground(claims: list[Any], ctx: RationaleContext) -> tuple[list[RationaleClaim], list[str]]:
        """Validate citations; ungrounded claims are quarantined, not trusted."""
        allowed = ctx.allowed_refs()
        grounded: list[RationaleClaim] = []
        unsupported: list[str] = []
        for item in claims:
            if not isinstance(item, dict):
                unsupported.append(repr(item))
                continue
            text = str(item.get("text") or "").strip()
            label_raw = str(item.get("label") or "UNKNOWN").upper()
            refs = [str(r) for r in (item.get("evidence_refs") or [])]
            if not text:
                unsupported.append("(empty claim text)")
                continue
            try:
                label = ClaimLabel(label_raw)
            except ValueError:
                label = ClaimLabel.UNKNOWN
                unsupported.append(f"{text[:60]}: invalid label {item.get('label')!r}")
            bad_refs = [r for r in refs if not _REF_PATTERN.match(r) or r not in allowed]
            if label is ClaimLabel.UNKNOWN and not refs:
                # explicitly-unknown claims need no citation
                grounded.append(RationaleClaim(text=text, label=label, evidence_refs=[]))
                continue
            if bad_refs:
                unsupported.append(f"{text[:60]}: unresolvable refs {bad_refs}")
                continue
            if label is not ClaimLabel.UNKNOWN and not refs:
                unsupported.append(f"{text[:60]}: factual claim without citation")
                continue
            grounded.append(RationaleClaim(text=text, label=label, evidence_refs=refs))
        return grounded, unsupported

    @staticmethod
    def _deterministic_sections(ctx: RationaleContext) -> dict[str, str]:
        """Non-LLM composition of the five sections from the packet itself."""

        def joined(lines: list[str], limit: int = 3) -> str:
            return " ".join(lines[:limit]) if lines else "No information recorded."

        decision = joined(ctx.requirement_matching)
        deviation = joined(
            [line for line in ctx.deviations if not line.startswith("relation")] or ctx.deviations
        )
        counterfactual = joined(ctx.counterfactuals)
        evidence = (
            f"{len(ctx.evidence)} evidence item(s) indexed "
            f"({', '.join(e.ref for e in ctx.evidence[:5])}"
            f"{'…' if len(ctx.evidence) > 5 else ''}); "
            + joined([e.ref_line() for e in ctx.evidence], limit=2)
        )
        uncertainty = joined(ctx.uncertainty + [f"missing: {m}" for m in ctx.missing])
        return {
            "decision_rationale": decision,
            "deviation_explanation": deviation,
            "counterfactual_summary": counterfactual,
            "evidence_summary": evidence,
            "uncertainty_statement": uncertainty,
        }

    @staticmethod
    def _deterministic_claims(ctx: RationaleContext) -> list[RationaleClaim]:
        """Grounded-by-construction claims for the fallback path."""
        claims: list[RationaleClaim] = []
        for assessment_line in ctx.requirement_matching[:3]:
            refs = _domain_refs(assessment_line, ctx)
            claims.append(RationaleClaim(
                text=assessment_line, label=ClaimLabel.EVIDENCE, evidence_refs=refs,
            ))
        if ctx.counterfactuals:
            claims.append(RationaleClaim(
                text=ctx.counterfactuals[0],
                label=ClaimLabel.EVIDENCE,
                evidence_refs=_domain_refs(ctx.counterfactuals[0], ctx),
            ))
        if ctx.missing:
            claims.append(RationaleClaim(
                text="; ".join(ctx.missing), label=ClaimLabel.UNKNOWN, evidence_refs=[],
            ))
        return claims

    def _fallback(self, ctx: RationaleContext, notes: list[str]) -> StructuredRationale:
        sections = self._deterministic_sections(ctx)
        return StructuredRationale(
            task_id=ctx.task_id, trace_id=ctx.trace_id, agent_id=ctx.agent_id,
            action=self._action_label(ctx),
            decision_rationale=sections["decision_rationale"],
            deviation_explanation=sections["deviation_explanation"],
            counterfactual_summary=sections["counterfactual_summary"],
            evidence_summary=sections["evidence_summary"],
            uncertainty_statement=sections["uncertainty_statement"],
            claims=self._deterministic_claims(ctx),
            unsupported_claims=[],
            grounded=True,
            provider=self._provider.name if self._provider else "deterministic",
            llm_used=False,
            provenance=self._provenance(ctx),
            notes=notes,
        )

    @staticmethod
    def _action_label(ctx: RationaleContext) -> str:
        for line in ctx.agent_action:
            if line.startswith("action:"):
                return line.removeprefix("action:").split(" via ")[0].strip()
        return "UNKNOWN"

    @staticmethod
    def _provenance(ctx: RationaleContext) -> dict:
        return {
            "synthetic_data": ctx.synthetic,
            "evidence_layers": ["Q1-decision", "Q2-deviation", "Q3-counterfactual"],
            "explain_layers_used_llm": False,
            "generator": RationaleGenerator.name,
            **ctx.provenance,
        }


def _domain_refs(line: str, ctx: RationaleContext) -> list[str]:
    """Refs present in a context line that are citable (E-numbers/REQ/ANOM/CF)."""
    allowed = ctx.allowed_refs()
    return sorted(
        ref for ref in allowed
        if re.search(rf"(?<![A-Za-z0-9-]){re.escape(ref)}(?![A-Za-z0-9-])", line)
    )


def generate_rationale(
    trace: ExecutionTrace, provider: RationaleProvider | None = None, **kwargs: Any
) -> StructuredRationale:
    """One-call convenience API."""
    return RationaleGenerator(provider=provider).generate(trace, **kwargs)
