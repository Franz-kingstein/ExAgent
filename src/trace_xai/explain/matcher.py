"""Requirement-to-action matcher (deterministic, auditable).

Algorithm
---------
1. Extract observable facts from the trace only (tool inputs, tool outputs,
   recorded state summaries). No hidden reasoning, no LLM.
2. For each applicable requirement, collect *candidate observations*:
   a. deterministic comparison when the operands are present in evidence
      (amount <= limit, verified == true, balance >= amount);
   b. the environment's reported outcome (reason_code), which is authoritative
      evidence for exactly the constraint it names.
3. Merge candidates:
   - no candidates      -> UNKNOWN (missing evidence is never assumed satisfied)
   - one distinct status -> that status; confidence 0.95 if any candidate was a
      deterministic comparison, else 0.9 (outcome-based)
   - disagreeing statuses -> UNKNOWN with all evidence listed as conflicting
4. Every assessment explanation cites its evidence as EVENT_ID.FIELD=VALUE so
   a reviewer can audit each claim against the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.trace import EventType, ExecutionTrace, TraceEvent
from ..requirements.model import ConstraintType, Requirement
from .models import EvidenceItem, RequirementAssessment, RequirementStatus

_SAT = RequirementStatus.SATISFIED
_VIOL = RequirementStatus.VIOLATED
_UNK = RequirementStatus.UNKNOWN

CONF_DETERMINISTIC = 0.95
CONF_OUTCOME = 0.9
CONF_NONE = 0.0


@dataclass
class TransferFacts:
    """Facts observable in a trace (None = genuinely not observed)."""

    action: str | None = None
    amount_inr: int | None = None
    beneficiary_id: str | None = None
    account_id: str | None = None
    limit_inr: int | None = None
    balance_before: int | None = None
    verified_in_state: bool | None = None
    invocation: TraceEvent | None = None
    transfer_results: list[TraceEvent] = field(default_factory=list)
    check_results: list[TraceEvent] = field(default_factory=list)


@dataclass
class _Observation:
    status: RequirementStatus
    evidence: list[EvidenceItem]
    basis: str
    deterministic: bool = False


def extract_facts(trace: ExecutionTrace) -> TransferFacts:
    """Pull observable facts out of a trace. Missing data stays None."""
    facts = TransferFacts()
    state_before: dict = {}

    for event in trace.events:
        if event.event_type is EventType.TOOL_INVOCATION:
            if facts.action is None:
                facts.action = event.action or event.tool_name
            if facts.invocation is None and event.tool_name == "transfer":
                facts.invocation = event
                facts.action = event.action or "transfer"
                tool_input = event.tool_input or {}
                facts.amount_inr = tool_input.get("amount_inr")
                facts.beneficiary_id = tool_input.get("beneficiary_id")
                if isinstance(event.state_before, dict):
                    state_before = event.state_before
        elif event.event_type is EventType.TOOL_RESULT:
            if event.tool_name == "transfer":
                facts.transfer_results.append(event)
                output = event.tool_output or {}
                # account id is observable in the result even when the input
                # relied on the environment's default account
                if facts.account_id is None:
                    facts.account_id = output.get("account_id")
            elif event.tool_name == "check_beneficiary":
                facts.check_results.append(event)

    if facts.account_id is None and facts.invocation is not None:
        facts.account_id = (facts.invocation.tool_input or {}).get("account_id")

    if state_before:
        facts.limit_inr = (state_before.get("per_txn_limits") or {}).get(facts.account_id)
        facts.balance_before = (state_before.get("balances") or {}).get(facts.account_id)
        verified_map = state_before.get("beneficiaries_verified") or {}
        if facts.beneficiary_id in verified_map:
            facts.verified_in_state = verified_map[facts.beneficiary_id]
    return facts


class RequirementMatcher:
    """Evaluates one requirement against one trace's observable facts."""

    def evaluate(
        self, trace: ExecutionTrace, facts: TransferFacts, requirement: Requirement
    ) -> RequirementAssessment:
        action = facts.action or "UNKNOWN"
        observations = self._observations(facts, requirement.constraint.type)
        return self._merge(requirement, action, facts, observations)

    # ---------------------------------------------------------- candidates
    def _observations(
        self, facts: TransferFacts, ctype: ConstraintType
    ) -> list[_Observation]:
        if ctype is ConstraintType.AMOUNT_WITHIN_LIMIT:
            return self._amount_observations(facts)
        if ctype is ConstraintType.BENEFICIARY_VERIFIED:
            return self._verified_observations(facts)
        if ctype is ConstraintType.SUFFICIENT_BALANCE:
            return self._balance_observations(facts)
        raise ValueError(f"Unsupported constraint: {ctype}")

    @staticmethod
    def _amount_observations(facts: TransferFacts) -> list[_Observation]:
        obs: list[_Observation] = []
        if facts.amount_inr is not None and facts.limit_inr is not None:
            satisfied = facts.amount_inr <= facts.limit_inr
            inv = facts.invocation
            obs.append(_Observation(
                status=_SAT if satisfied else _VIOL,
                evidence=[
                    EvidenceItem(event_id=inv.event_id, field="tool_input.amount_inr",
                                 value=facts.amount_inr),
                    EvidenceItem(event_id=inv.event_id,
                                 field=f"state_before.per_txn_limits.{facts.account_id}",
                                 value=facts.limit_inr),
                ],
                basis=(f"amount {facts.amount_inr} {'<=' if satisfied else '>'} "
                       f"limit {facts.limit_inr}"),
                deterministic=True,
            ))
        for result in facts.transfer_results:
            code = (result.tool_output or {}).get("reason_code")
            if code == "OK":
                obs.append(_Observation(
                    _SAT,
                    [EvidenceItem(event_id=result.event_id,
                                  field="tool_output.reason_code", value=code)],
                    "transfer reported success (tool contract: fails when over limit)",
                ))
            elif code == "LIMIT_EXCEEDED":
                obs.append(_Observation(
                    _VIOL,
                    [EvidenceItem(event_id=result.event_id,
                                  field="tool_output.reason_code", value=code)],
                    "transfer reported LIMIT_EXCEEDED",
                ))
        return obs

    @staticmethod
    def _verified_observations(facts: TransferFacts) -> list[_Observation]:
        obs: list[_Observation] = []
        if facts.beneficiary_id is not None and facts.verified_in_state is not None:
            inv = facts.invocation
            obs.append(_Observation(
                status=_SAT if facts.verified_in_state else _VIOL,
                evidence=[EvidenceItem(
                    event_id=inv.event_id,
                    field=f"state_before.beneficiaries_verified.{facts.beneficiary_id}",
                    value=facts.verified_in_state)],
                basis=f"recorded verification status of {facts.beneficiary_id} is "
                      f"{facts.verified_in_state}",
                deterministic=True,
            ))
        for result in facts.check_results:
            verified = (result.tool_output or {}).get("verified")
            if isinstance(verified, bool):
                obs.append(_Observation(
                    status=_SAT if verified else _VIOL,
                    evidence=[EvidenceItem(event_id=result.event_id,
                                           field="tool_output.verified", value=verified)],
                    basis=f"check_beneficiary reported verified={verified}",
                ))
        for result in facts.transfer_results:
            code = (result.tool_output or {}).get("reason_code")
            if code == "OK":
                obs.append(_Observation(
                    _SAT,
                    [EvidenceItem(event_id=result.event_id,
                                  field="tool_output.reason_code", value=code)],
                    "transfer reported success (tool contract: fails if unverified)",
                ))
            elif code == "BENEFICIARY_UNVERIFIED":
                obs.append(_Observation(
                    _VIOL,
                    [EvidenceItem(event_id=result.event_id,
                                  field="tool_output.reason_code", value=code)],
                    "transfer reported BENEFICIARY_UNVERIFIED",
                ))
        return obs

    @staticmethod
    def _balance_observations(facts: TransferFacts) -> list[_Observation]:
        obs: list[_Observation] = []
        if facts.amount_inr is not None and facts.balance_before is not None:
            satisfied = facts.balance_before >= facts.amount_inr
            inv = facts.invocation
            obs.append(_Observation(
                status=_SAT if satisfied else _VIOL,
                evidence=[
                    EvidenceItem(event_id=inv.event_id,
                                 field=f"state_before.balances.{facts.account_id}",
                                 value=facts.balance_before),
                    EvidenceItem(event_id=inv.event_id,
                                 field="tool_input.amount_inr", value=facts.amount_inr),
                ],
                basis=(f"balance {facts.balance_before} {'>=' if satisfied else '<'} "
                       f"amount {facts.amount_inr}"),
                deterministic=True,
            ))
        for result in facts.transfer_results:
            code = (result.tool_output or {}).get("reason_code")
            if code == "OK":
                obs.append(_Observation(
                    _SAT,
                    [EvidenceItem(event_id=result.event_id,
                                  field="tool_output.reason_code", value=code)],
                    "transfer reported success (tool contract: fails on insufficient funds)",
                ))
            elif code == "INSUFFICIENT_FUNDS":
                obs.append(_Observation(
                    _VIOL,
                    [EvidenceItem(event_id=result.event_id,
                                  field="tool_output.reason_code", value=code)],
                    "transfer reported INSUFFICIENT_FUNDS",
                ))
        return obs

    # -------------------------------------------------------------- merge
    def _merge(
        self,
        requirement: Requirement,
        action: str,
        facts: TransferFacts,
        observations: list[_Observation],
    ) -> RequirementAssessment:
        statuses = {o.status for o in observations}

        if not statuses:
            return RequirementAssessment(
                requirement_id=requirement.id,
                requirement_text=requirement.text,
                relevant_action=action,
                status=_UNK,
                explanation="No observable evidence in the trace for this requirement.",
                confidence=CONF_NONE,
            )

        if len(statuses) > 1:
            conflicting = [e for o in observations for e in o.evidence]
            detail = "; ".join(
                f"{o.basis} -> {o.status.value}" for o in observations
            )
            return RequirementAssessment(
                requirement_id=requirement.id,
                requirement_text=requirement.text,
                relevant_action=action,
                status=_UNK,
                conflicting_evidence=conflicting,
                explanation=f"Conflicting evidence ({detail}). No status assigned.",
                confidence=CONF_NONE,
            )

        status = statuses.pop()
        supporting = self._dedupe([e for o in observations for e in o.evidence])
        confidence = (
            CONF_DETERMINISTIC if any(o.deterministic for o in observations) else CONF_OUTCOME
        )
        citations = ", ".join(f"{e.event_id}.{e.field}={e.value!r}" for e in supporting)
        clause = _clause(requirement.constraint.type, status, facts)
        return RequirementAssessment(
            requirement_id=requirement.id,
            requirement_text=requirement.text,
            relevant_action=action,
            status=status,
            supporting_evidence=supporting,
            explanation=f"{clause} Evidence: {citations}.",
            confidence=confidence,
        )

    @staticmethod
    def _dedupe(items: list[EvidenceItem]) -> list[EvidenceItem]:
        seen: set[tuple[str, str]] = set()
        unique: list[EvidenceItem] = []
        for item in items:
            key = (item.event_id, item.field)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique


def _clause(ctype: ConstraintType, status: RequirementStatus, facts: TransferFacts) -> str:
    """Concise, human-readable sentence for one requirement verdict.

    None-safe by design: a verdict may come from the recorded outcome while
    the operand value itself was not observable — the clause says so instead
    of inventing numbers.
    """
    if status is _UNK:
        return f"Evidence was missing or conflicting for {ctype.value}."
    ok = status is _SAT
    if ctype is ConstraintType.AMOUNT_WITHIN_LIMIT:
        if facts.limit_inr is not None:
            return (
                f"The transfer amount (INR {facts.amount_inr}) was "
                f"{'within' if ok else 'above'} the per-transaction limit "
                f"(INR {facts.limit_inr})."
            )
        return (
            f"The recorded transfer outcome indicates the amount was "
            f"{'within' if ok else 'above'} the applicable limit "
            f"(limit value not observable in the trace)."
        )
    if ctype is ConstraintType.BENEFICIARY_VERIFIED:
        who = f" ({facts.beneficiary_id})" if facts.beneficiary_id else ""
        return (
            f"The beneficiary{who} was {'verified' if ok else 'not verified'} "
            f"according to the recorded verification result."
        )
    if facts.balance_before is not None:
        return (
            f"The account ({facts.account_id}) balance (INR {facts.balance_before}) was "
            f"{'sufficient' if ok else 'insufficient'} for the transfer "
            f"(INR {facts.amount_inr})."
        )
    return (
        f"The recorded transfer outcome indicates the balance was "
        f"{'sufficient' if ok else 'insufficient'} "
        f"(balance not observable in the trace)."
    )
