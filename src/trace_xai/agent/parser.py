"""Deterministic natural-language task parser.

Converts tasks like "Transfer ₹10,000 to beneficiary BOB-001" into a typed
Intent. Pure rules, no LLM: this is the deterministic front door used by the
rule-based agent and available to any other agent.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    BALANCE = "BALANCE"
    BENEFICIARY_CHECK = "BENEFICIARY_CHECK"
    HISTORY = "HISTORY"
    TRANSFER = "TRANSFER"
    UNKNOWN = "UNKNOWN"


class Intent(BaseModel):
    type: IntentType
    raw_text: str
    account_id: str | None = None
    beneficiary_id: str | None = None
    amount_inr: int | None = None
    note: str | None = Field(default=None, description="Parse diagnostics (e.g. bad amount)")


_ACCOUNT_RE = re.compile(r"(ACC-SYN-\d+)", re.IGNORECASE)
_BENEFICIARY_RE = re.compile(r"\b([A-Z]{2,6}-\d{2,4})\b")

# "Transfer ₹10,000 to beneficiary BOB-001" / "transfer 10000 to BOB-001"
_TRANSFER_RE = re.compile(
    r"\btransfer\b\s*(?:of\s*)?(?:₹|rs\.?\s*|inr\s*)?\s*([\d,]+(?:\.\d+)?)"
    r".*?\bto\s+(?:beneficiary\s+)?([A-Z]{2,6}-\d{2,4})\b",
    re.IGNORECASE | re.DOTALL,
)
_HISTORY_RE = re.compile(r"\b(history|transactions|statement)\b", re.IGNORECASE)
_BENEFICIARY_CHECK_RE = re.compile(
    r"\bbeneficiary\b.*\b(verify|verified|verification|check|status)\b|\b"
    r"(verify|verify|check)\s+(?:the\s+)?beneficiary\b",
    re.IGNORECASE,
)
_BALANCE_RE = re.compile(r"\b(balance|funds)\b", re.IGNORECASE)


def _parse_amount(text: str) -> int | None:
    """Parse an integer INR amount; returns None for malformed/non-integer."""
    cleaned = text.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def parse_task(text: str) -> Intent:
    """Parse a natural-language banking task into a typed Intent."""
    raw = text.strip()

    m = _TRANSFER_RE.search(raw)
    if m:
        amount = _parse_amount(m.group(1))
        note = None
        if amount is None:
            note = f"Malformed amount: {m.group(1)!r} (integer INR required)"
        return Intent(
            type=IntentType.TRANSFER,
            raw_text=raw,
            account_id=_account(raw),
            beneficiary_id=m.group(2).upper(),
            amount_inr=amount,
            note=note,
        )

    if _BENEFICIARY_CHECK_RE.search(raw):
        b = _BENEFICIARY_RE.search(raw)
        return Intent(
            type=IntentType.BENEFICIARY_CHECK,
            raw_text=raw,
            account_id=_account(raw),
            beneficiary_id=b.group(1).upper() if b else None,
        )

    if _HISTORY_RE.search(raw):
        return Intent(type=IntentType.HISTORY, raw_text=raw, account_id=_account(raw))

    if _BALANCE_RE.search(raw):
        return Intent(type=IntentType.BALANCE, raw_text=raw, account_id=_account(raw))

    return Intent(type=IntentType.UNKNOWN, raw_text=raw, note="No supported intent matched")


def _account(text: str) -> str | None:
    m = _ACCOUNT_RE.search(text)
    return m.group(1).upper() if m else None
