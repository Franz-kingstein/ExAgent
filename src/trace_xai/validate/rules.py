"""Deterministic claim verification over structured evidence.

Preferred over any model judge: these rules compare the claim's asserted
relation/value against the *cited* evidence operands only. A rule fires only
when it can decide from the cited evidence; otherwise it returns None and the
pipeline falls through (lexical NLI backstop -> LLM judge -> UNKNOWN).

Rule inventory (first decisive rule wins):
  1. limit relation   "within/under the limit" / "exceeded/over the limit"
  2. balance relation "sufficient funds" / "insufficient funds"
  3. verification     "beneficiary (un)verified"      (missing -> UNKNOWN)
  4. outcome          "transfer succeeded/rejected" + reason-code equality
                      (conflicting success values in evidence -> UNKNOWN)
  5. requirement      "REQ-X was satisfied/violated" vs recorded status
  6. asserted number  "balance was 9,999" vs recorded operand value
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import VerificationMethod, VerificationStatus

_SUP = VerificationStatus.SUPPORTED
_CON = VerificationStatus.CONTRADICTED
_UNK = VerificationStatus.UNKNOWN


@dataclass
class RuleHit:
    status: VerificationStatus
    explanation: str
    confidence: float


@dataclass
class Operands:
    """Decision-relevant values extracted from the *cited* evidence."""

    amounts: list[int] = field(default_factory=list)
    limits: list[int] = field(default_factory=list)
    balances: list[int] = field(default_factory=list)
    verified: list[bool] = field(default_factory=list)   # known verification flags
    verification_cited: bool = False                     # any verification evidence cited
    success_values: list[bool] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)


_NUM = r"(\d[\d,]*(?:\.\d+)?)"


def _to_int(raw: str) -> int:
    return int(float(raw.replace(",", "")))


def extract_operands(evidence_values: list[tuple[str, object]]) -> Operands:
    """(field, value) pairs cited by the claim -> structured operands."""
    ops = Operands()
    for field_name, value in evidence_values:
        low = field_name.lower()
        if "per_txn_limits" in low and isinstance(value, (int, float)):
            ops.limits.append(int(value))
        elif "balances" in low and isinstance(value, (int, float)):
            ops.balances.append(int(value))
        elif "beneficiaries_verified" in low:
            ops.verification_cited = True
            if isinstance(value, bool):
                ops.verified.append(value)
        elif low.endswith("amount_inr") and isinstance(value, (int, float)):
            ops.amounts.append(int(value))
        elif low.endswith("reason_code") and isinstance(value, str):
            ops.reason_codes.append(value)
        elif low.endswith("success") and isinstance(value, bool):
            ops.success_values.append(value)
    return ops


# --------------------------------------------------------------- rules
_WITHIN = re.compile(
    r"\b(within|under|below|inside)\b[^.]{0,40}\blimit\b"
    r"|\bdid not exceed\b|\bnot exceed(?:ed)?\b|\bcompliant with\b", re.I)
_EXCEEDED = re.compile(
    r"\b(exceed(?:ed|s)?|over the limit|surpass(?:ed|es)?|above the limit)\b", re.I)
_INSUFFICIENT = re.compile(r"\binsufficient\b|\bnot enough funds?\b", re.I)
_SUFFICIENT = re.compile(r"\b(sufficient|adequate)\b[^.]{0,30}\b(balance|funds?)\b", re.I)
_VERIFIED = re.compile(r"\b(beneficiary|payee)\b[^.]{0,30}\b(was|is|been)\s+verified\b", re.I)
_UNVERIFIED = re.compile(
    r"\b(beneficiary|payee)\b[^.]{0,40}\b(unverified|not verified|never verified)\b", re.I)
_REJECTED = re.compile(r"\b(rejected|declined|refused|failed)\b", re.I)
_SUCCEEDED = re.compile(
    r"\b(succeeded|was successful|completed|went through|approved)\b", re.I)
_REASON_CODE = re.compile(r"\b[A-Z][A-Z_]{3,}\b")
_REQ_ID = re.compile(r"\bREQ-[A-Z]+-\d+\b")
_REQ_VIOLATED = re.compile(r"\b(violat(?:ed|ion|es)|breach(?:ed)?)\b", re.I)
_REQ_SATISFIED = re.compile(r"\b(satisf(?:ied|ies|y)|compliant|met)\b", re.I)
_ASSERT_NUM = re.compile(
    rf"\b(balance|limit|amount)\b\w*(?:\s+\w+){{0,3}}?\s+"
    rf"(?:was|is|of|equals?|=)\s+(?:inr\s*)?{_NUM}", re.I)


def check_limit(claim: str, ops: Operands) -> RuleHit | None:
    if not (ops.amounts and ops.limits):
        return None
    amount, limit = max(ops.amounts), max(ops.limits)
    if _WITHIN.search(claim):
        ok = amount <= limit
        return RuleHit(_SUP if ok else _CON,
                       f"claimed within limit; cited amount {amount} vs limit {limit} "
                       f"-> {'within' if ok else 'NOT within'}", 0.95)
    if _EXCEEDED.search(claim) and re.search(r"limit|amount", claim, re.I):
        ok = amount > limit
        return RuleHit(_SUP if ok else _CON,
                       f"claimed limit exceeded; cited amount {amount} vs limit {limit} "
                       f"-> {'exceeded' if ok else 'NOT exceeded'}", 0.95)
    return None


def check_balance(claim: str, ops: Operands) -> RuleHit | None:
    if not (ops.amounts and ops.balances):
        return None
    amount, balance = max(ops.amounts), min(ops.balances)
    if _INSUFFICIENT.search(claim):
        ok = balance < amount
        return RuleHit(_SUP if ok else _CON,
                       f"claimed insufficient funds; cited balance {balance} vs amount "
                       f"{amount} -> {'insufficient' if ok else 'sufficient'}", 0.95)
    if _SUFFICIENT.search(claim):
        ok = balance >= amount
        return RuleHit(_SUP if ok else _CON,
                       f"claimed sufficient funds; cited balance {balance} vs amount "
                       f"{amount} -> {'sufficient' if ok else 'insufficient'}", 0.95)
    return None


def check_verification(claim: str, ops: Operands) -> RuleHit | None:
    positive = _VERIFIED.search(claim)
    negative = _UNVERIFIED.search(claim)
    if not (positive or negative):
        return None
    if not ops.verification_cited or not ops.verified:
        # spec example 1: verification UNKNOWN must never come back SUPPORTED
        return RuleHit(_UNK,
                       "claim asserts beneficiary verification but the cited evidence "
                       "contains no (usable) verification value -> UNKNOWN", 0.2)
    all_true = all(ops.verified)
    any_false = not all(verified is True for verified in ops.verified)
    if positive:
        if any_false:
            return RuleHit(_CON, f"cited verification value is {ops.verified} "
                                 f"(not verified) but claim says verified", 0.95)
        return RuleHit(_SUP, f"cited verification value {ops.verified} confirms verified",
                       0.95)
    # negative phrasing
    if all_true:
        return RuleHit(_CON, f"cited verification value {ops.verified} confirms verified "
                             f"but claim says unverified", 0.95)
    return RuleHit(_SUP, f"cited verification value {ops.verified} confirms unverified",
                   0.95)


def check_outcome(claim: str, ops: Operands) -> RuleHit | None:
    code = None
    for token in _REASON_CODE.findall(claim):
        if ops.reason_codes:
            code = token
            break
    negative = _REJECTED.search(claim)
    positive = _SUCCEEDED.search(claim)
    if code:
        match = code in ops.reason_codes
        return RuleHit(_SUP if match else _CON,
                       f"claimed reason {code}; cited evidence reason_codes="
                       f"{ops.reason_codes} -> {'match' if match else 'MISMATCH'}", 0.95)
    if not (negative or positive):
        return None
    if len(set(ops.success_values)) > 1:
        return RuleHit(_UNK,
                       f"evidence contains conflicting success values "
                       f"{ops.success_values} -> conflict, cannot decide", 0.4)
    if not ops.success_values:
        return None
    success = ops.success_values[0]
    if positive:
        return RuleHit(_SUP if success else _CON,
                       f"claimed success; cited success={success}", 0.95)
    return RuleHit(_SUP if not success else _CON,
                   f"claimed rejection/failure; cited success={success}", 0.95)


def check_requirement_status(claim: str, req_statuses: dict[str, str]) -> RuleHit | None:
    req_ids = _REQ_ID.findall(claim)
    if not req_ids:
        return None
    violated_word = _REQ_VIOLATED.search(claim)
    satisfied_word = _REQ_SATISFIED.search(claim)
    if not (violated_word or satisfied_word):
        return None
    for req_id in req_ids:
        recorded = req_statuses.get(req_id)
        if recorded is None:
            continue
        recorded = recorded.upper()
        if "VIOLATED" in recorded and violated_word:
            return RuleHit(_SUP, f"{req_id} recorded as {recorded}; claim agrees", 0.95)
        if "VIOLATED" in recorded and satisfied_word:
            return RuleHit(_CON, f"{req_id} recorded as {recorded} but claim says "
                                 f"satisfied/compliant", 0.95)
        if "SATISFIED" in recorded and satisfied_word:
            return RuleHit(_SUP, f"{req_id} recorded as {recorded}; claim agrees", 0.95)
        if "SATISFIED" in recorded and violated_word:
            return RuleHit(_CON, f"{req_id} recorded as {recorded} but claim says "
                                 f"violated", 0.95)
    return None


def check_asserted_number(claim: str, ops: Operands) -> RuleHit | None:
    """'balance was 9,999' style assertions against recorded operand values."""
    classes = {"balance": ops.balances, "limit": ops.limits, "amount": ops.amounts}
    for match in _ASSERT_NUM.finditer(claim):
        noun = match.group(1).lower()
        values = classes.get(noun) or []
        if not values:
            continue
        asserted = _to_int(match.group(2))
        if asserted in values:
            return RuleHit(_SUP, f"asserted {noun} {asserted} matches cited evidence "
                                 f"{values}", 0.9)
        return RuleHit(_CON, f"asserted {noun} {asserted} conflicts with cited "
                             f"evidence {values}", 0.9)
    return None


def run_rules(claim: str, ops: Operands, req_statuses: dict[str, str]) -> RuleHit | None:
    for hit in (
        check_limit(claim, ops),
        check_balance(claim, ops),
        check_verification(claim, ops),
        check_outcome(claim, ops),
        check_requirement_status(claim, req_statuses),
        check_asserted_number(claim, ops),
    ):
        if hit is not None:
            return hit
    return None


# ---------------------------------------------------------- lexical NLI
_NEGATION = re.compile(
    r"\b(not|never|no|cannot|wasn't|isn't|weren't|aren't|didn't|doesn't|won't|"
    r"lacks?|without|insufficient|unverified|rejected|failed|exceeded)\b", re.I)
_ANTONYM_PAIRS = (
    ("verified", "unverified"), ("sufficient", "insufficient"),
    ("succeeded", "failed"), ("within", "exceeded"), ("allowed", "blocked"),
)


def lexical_entailment(claim: str, evidence_text: str) -> RuleHit | None:
    """Backstop only: detects obvious negation/antonym clashes.

    Never marks a claim SUPPORTED (too weak for that); it can only turn an
    undecided claim into CONTRADICTED when a negation or antonym directly
    clashes with the cited evidence wording.
    """
    claim_low, ev_low = claim.lower(), evidence_text.lower()
    claim_neg = bool(_NEGATION.search(claim_low))
    ev_neg = bool(_NEGATION.search(ev_low))
    if claim_neg != ev_neg:
        # negation mismatch is only decisive when the topics overlap
        overlap = _content_overlap(claim_low, ev_low)
        if overlap >= 0.4:
            return RuleHit(_CON,
                           f"negation mismatch (claim negated={claim_neg}, evidence "
                           f"negated={ev_neg}) with lexical overlap {overlap:.2f}",
                           0.6)
    for strong, weak in _ANTONYM_PAIRS:
        if strong in claim_low and weak in ev_low:
            return RuleHit(_CON, f"claim asserts '{strong}' but cited evidence says "
                                 f"'{weak}'", 0.6)
        if weak in claim_low and strong in ev_low:
            return RuleHit(_CON, f"claim asserts '{weak}' but cited evidence says "
                                 f"'{strong}'", 0.6)
    return None


def content_overlap(claim: str, evidence_text: str) -> float:
    return _content_overlap(claim.lower(), evidence_text.lower())


def _content_overlap(claim_low: str, ev_low: str) -> float:
    stop = {"the", "a", "an", "was", "were", "is", "are", "of", "for", "to", "in",
            "and", "or", "that", "this", "with", "by", "it", "be", "has", "have"}
    claim_tokens = {t for t in re.findall(r"[a-z]+", claim_low) if t not in stop}
    if not claim_tokens:
        return 0.0
    ev_tokens = set(re.findall(r"[a-z]+", ev_low))
    return len(claim_tokens & ev_tokens) / len(claim_tokens)
