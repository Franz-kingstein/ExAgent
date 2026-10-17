"""Lightweight, interpretable trajectory anomaly detector (rule-based).

No ML model: every detector is an explicit, auditable rule over observable
trace events. Signals covered:

- unexpected tool call            -> UNEXPECTED_TOOL_CALL
- unexpected event order          -> UNEXPECTED_EVENT_ORDER
- missing expected event          -> MISSING_EXPECTED_EVENT (no outcome recorded)
- conflicting tool results        -> CONFLICTING_TOOL_RESULTS
- repeated failed actions         -> REPEATED_FAILED_ACTION (>= 2 identical failures)
- excessive retries               -> EXCESSIVE_RETRY (>= 3 identical failures)
- inconsistent state transition   -> INCONSISTENT_STATE_TRANSITION
- action inconsistent with state  -> ACTION_INCONSISTENT_WITH_STATE

Every anomaly cites the exact events and values it was derived from. Traces
with missing state (e.g. LLM traces) are skipped by state checks rather than
flagged — absence of evidence is not evidence of anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.trace import EventType, ExecutionTrace, TraceEvent
from .models import Anomaly, AnomalyType, EvidenceItem, Severity

_SAT = True
_FAIL = False


@dataclass
class _Finding:
    """Internal anomaly candidate; gets its final id after sorting."""

    event_ids: list[str]
    anomaly_type: AnomalyType
    severity: Severity
    explanation: str
    confidence: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    min_sequence: int = 0


class TrajectoryAnomalyDetector:
    """Deterministic, rule-based anomaly detector over ExecutionTraces."""

    name = "trajectory-anomaly-detector-v1"

    ALLOWED_TOOLS = frozenset(
        {"get_balance", "check_beneficiary", "get_history", "transfer"}
    )
    RETRY_FAILURE_THRESHOLD = 2
    EXCESSIVE_RETRY_THRESHOLD = 3

    def detect(self, trace: ExecutionTrace) -> list[Anomaly]:
        events_by_id = {e.event_id: e for e in trace.events}
        invocations = [e for e in trace.events if e.event_type is EventType.TOOL_INVOCATION]
        results = [e for e in trace.events if e.event_type is EventType.TOOL_RESULT]
        outcomes = [e for e in trace.events if e.event_type is EventType.OUTCOME_RECORDED]

        findings: list[_Finding] = []
        self._check_unexpected_tools(invocations + results, events_by_id, findings)
        self._check_ordering(trace, invocations, results, outcomes, events_by_id, findings)
        self._check_conflicts(results, events_by_id, findings)
        self._check_retries(invocations, results, events_by_id, findings)
        self._check_state_consistency(invocations, results, events_by_id, findings)
        self._check_action_vs_state(invocations, results, events_by_id, findings)

        findings.sort(key=lambda f: (f.min_sequence, f.anomaly_type.value, f.explanation))
        return [
            Anomaly(
                anomaly_id=f"ANOM-{n:04d}",
                event_ids=f.event_ids,
                anomaly_type=f.anomaly_type,
                evidence=f.evidence,
                severity=f.severity,
                explanation=f.explanation,
                confidence=f.confidence,
            )
            for n, f in enumerate(findings, start=1)
        ]

    # ---------------------------------------------------------------- checks
    def _check_unexpected_tools(self, events, events_by_id, findings) -> None:
        by_tool: dict[str, list[TraceEvent]] = {}
        for event in events:
            if event.tool_name and event.tool_name not in self.ALLOWED_TOOLS:
                by_tool.setdefault(event.tool_name, []).append(event)
        for tool_name, flagged in by_tool.items():
            findings.append(_Finding(
                event_ids=[e.event_id for e in flagged],
                anomaly_type=AnomalyType.UNEXPECTED_TOOL_CALL,
                severity=Severity.HIGH,
                confidence=1.0,
                explanation=f"Tool '{tool_name}' is not part of the expected "
                            f"banking toolset.",
                evidence=[EvidenceItem(event_id=e.event_id, field="tool_name",
                                       value=e.tool_name) for e in flagged],
                min_sequence=min(e.sequence for e in flagged),
            ))

    def _check_ordering(self, trace, invocations, results, outcomes,
                        events_by_id, findings) -> None:
        first_outcome = min((e.sequence for e in outcomes), default=None)

        for event in invocations + results:
            if first_outcome is not None and event.sequence > first_outcome:
                findings.append(_Finding(
                    event_ids=[event.event_id],
                    anomaly_type=AnomalyType.UNEXPECTED_EVENT_ORDER,
                    severity=Severity.MEDIUM,
                    confidence=1.0,
                    explanation=f"{event.event_type.value} occurs after the outcome "
                                f"was already recorded.",
                    evidence=[EvidenceItem(event_id=event.event_id, field="event_type",
                                           value=event.event_type.value)],
                    min_sequence=event.sequence,
                ))

        for result in results:
            ref = result.related_event_id
            if ref is None or ref not in events_by_id:
                findings.append(_Finding(
                    event_ids=[result.event_id],
                    anomaly_type=AnomalyType.UNEXPECTED_EVENT_ORDER,
                    severity=Severity.MEDIUM,
                    confidence=0.9,
                    explanation="Tool result has no resolvable invocation event "
                                "(dangling or missing reference).",
                    evidence=[EvidenceItem(event_id=result.event_id,
                                           field="related_event_id", value=ref)],
                    min_sequence=result.sequence,
                ))
            else:
                ref_event = events_by_id[ref]
                if (ref_event.sequence >= result.sequence
                        or ref_event.event_type is not EventType.TOOL_INVOCATION):
                    findings.append(_Finding(
                        event_ids=[result.event_id, ref],
                        anomaly_type=AnomalyType.UNEXPECTED_EVENT_ORDER,
                        severity=Severity.HIGH,
                        confidence=1.0,
                        explanation="Tool result references an event that is not its "
                                    "earlier invocation.",
                        evidence=[EvidenceItem(event_id=result.event_id,
                                               field="related_event_id", value=ref)],
                        min_sequence=min(ref_event.sequence, result.sequence),
                    ))

        if not outcomes:
            findings.append(_Finding(
                event_ids=[e.event_id for e in trace.events] or ["<trace>"],
                anomaly_type=AnomalyType.MISSING_EXPECTED_EVENT,
                severity=Severity.MEDIUM,
                confidence=1.0,
                explanation="Trace contains no OUTCOME_RECORDED event.",
                evidence=[],
                min_sequence=0,
            ))

    def _check_conflicts(self, results, events_by_id, findings) -> None:
        by_invocation: dict[str, list[TraceEvent]] = {}
        for result in results:
            if result.related_event_id:
                by_invocation.setdefault(result.related_event_id, []).append(result)
        for inv_id, group in by_invocation.items():
            if len(group) < 2:
                continue
            outcomes_seen = {
                (g.success, (g.tool_output or {}).get("reason_code")) for g in group
            }
            if len(outcomes_seen) > 1:
                findings.append(_Finding(
                    event_ids=[inv_id] + [g.event_id for g in group],
                    anomaly_type=AnomalyType.CONFLICTING_TOOL_RESULTS,
                    severity=Severity.HIGH,
                    confidence=1.0,
                    explanation=f"{len(group)} tool results for the same invocation "
                                f"report different outcomes "
                                f"({sorted(str(o) for o in outcomes_seen)}).",
                    evidence=[EvidenceItem(event_id=g.event_id,
                                           field="tool_output.reason_code",
                                           value=(g.tool_output or {}).get("reason_code"))
                              for g in group],
                    min_sequence=events_by_id[inv_id].sequence if inv_id in events_by_id
                    else min(g.sequence for g in group),
                ))

    def _check_retries(self, invocations, results, events_by_id, findings) -> None:
        groups: dict[tuple, list[str]] = {}
        for invocation in invocations:
            key = (
                invocation.tool_name,
                tuple(sorted((invocation.tool_input or {}).items())),
            )
            groups.setdefault(key, []).append(invocation.event_id)

        for (tool_name, _args), inv_ids in groups.items():
            failed = [
                r for r in results
                if r.related_event_id in inv_ids and r.success is _FAIL
            ]
            count = len(failed)
            if count < self.RETRY_FAILURE_THRESHOLD:
                continue
            excessive = count >= self.EXCESSIVE_RETRY_THRESHOLD
            reasons = [
                (r.tool_output or {}).get("reason_code", "?") for r in failed
            ]
            findings.append(_Finding(
                event_ids=inv_ids + [r.event_id for r in failed],
                anomaly_type=(AnomalyType.EXCESSIVE_RETRY if excessive
                              else AnomalyType.REPEATED_FAILED_ACTION),
                severity=Severity.HIGH if excessive else Severity.MEDIUM,
                confidence=0.9,
                explanation=f"Tool '{tool_name}' failed {count} time(s) for identical "
                            f"input (reasons: {reasons}).",
                evidence=[EvidenceItem(event_id=r.event_id, field="tool_output.reason_code",
                                       value=(r.tool_output or {}).get("reason_code"))
                          for r in failed],
                min_sequence=min(events_by_id[i].sequence for i in inv_ids
                                 if i in events_by_id) if any(
                    i in events_by_id for i in inv_ids) else min(r.sequence for r in failed),
            ))

    def _check_state_consistency(self, invocations, results, events_by_id, findings) -> None:
        results_by_invocation: dict[str, list[TraceEvent]] = {}
        for result in results:
            if result.related_event_id:
                results_by_invocation.setdefault(result.related_event_id, []).append(result)

        for invocation in invocations:
            if invocation.tool_name != "transfer":
                continue
            if not isinstance(invocation.state_before, dict):
                continue  # state unobservable (e.g. LLM trace): skip, do not flag
            before = invocation.state_before.get("balances") or {}
            args = invocation.tool_input or {}
            amount = args.get("amount_inr")

            for result in results_by_invocation.get(invocation.event_id, []):
                if not isinstance(result.state_after, dict):
                    continue
                after = result.state_after.get("balances") or {}
                output = result.tool_output or {}
                account_id = output.get("account_id") or args.get("account_id")

                if (result.success is _SAT and amount is not None
                        and account_id in before and account_id in after):
                    expected = before[account_id] - amount
                    if after[account_id] != expected:
                        findings.append(_Finding(
                            event_ids=[invocation.event_id, result.event_id],
                            anomaly_type=AnomalyType.INCONSISTENT_STATE_TRANSITION,
                            severity=Severity.HIGH,
                            confidence=1.0,
                            explanation=f"Successful transfer but recorded balance is "
                                        f"{after[account_id]}, expected {expected}.",
                            evidence=[
                                EvidenceItem(event_id=invocation.event_id,
                                             field=f"state_before.balances.{account_id}",
                                             value=before[account_id]),
                                EvidenceItem(event_id=result.event_id,
                                             field=f"state_after.balances.{account_id}",
                                             value=after[account_id]),
                            ],
                            min_sequence=invocation.sequence,
                        ))
                if result.success is _FAIL:
                    changed = sorted(
                        k for k in set(before) | set(after)
                        if before.get(k) != after.get(k)
                    )
                    if changed:
                        findings.append(_Finding(
                            event_ids=[invocation.event_id, result.event_id],
                            anomaly_type=AnomalyType.INCONSISTENT_STATE_TRANSITION,
                            severity=Severity.HIGH,
                            confidence=1.0,
                            explanation=f"Rejected transfer changed balances of "
                                        f"{changed} (funds moved despite failure).",
                            evidence=[EvidenceItem(event_id=result.event_id,
                                                   field="state_after.balances",
                                                   value={k: after.get(k) for k in changed})],
                            min_sequence=invocation.sequence,
                        ))

    def _check_action_vs_state(self, invocations, results, events_by_id, findings) -> None:
        unverified_at: dict[str, TraceEvent] = {}
        for result in results:
            if result.tool_name == "check_beneficiary":
                output = result.tool_output or {}
                if output.get("verified") is False:
                    beneficiary_id = output.get("beneficiary_id") or (
                        # older/compact results may omit it; fall back to null
                        None
                    )
                    if beneficiary_id:
                        unverified_at[beneficiary_id] = result

        for invocation in invocations:
            if invocation.tool_name != "transfer":
                continue
            beneficiary_id = (invocation.tool_input or {}).get("beneficiary_id")
            check = unverified_at.get(beneficiary_id)
            if check is not None and check.sequence < invocation.sequence:
                findings.append(_Finding(
                    event_ids=[check.event_id, invocation.event_id],
                    anomaly_type=AnomalyType.ACTION_INCONSISTENT_WITH_STATE,
                    severity=Severity.MEDIUM,
                    confidence=0.9,
                    explanation=f"Transfer attempted for beneficiary '{beneficiary_id}' "
                                f"after it was observed unverified.",
                    evidence=[
                        EvidenceItem(event_id=check.event_id,
                                     field="tool_output.verified", value=False),
                        EvidenceItem(event_id=invocation.event_id,
                                     field="tool_input.beneficiary_id",
                                     value=beneficiary_id),
                    ],
                    min_sequence=check.sequence,
                ))
