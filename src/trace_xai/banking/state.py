"""State/data models for the simulated bank (SYNTHETIC DATA ONLY).

These models represent the *state* layer. They contain no behavior beyond
validation, so the state can be inspected, snapshotted, and later fed into
the TRACE-XAI evidence store without touching the environment code.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    CLOSED = "CLOSED"


class TransactionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"


class Beneficiary(BaseModel):
    """An external payee linked to an account. All fields synthetic."""

    id: str = Field(pattern=r"^[A-Z]{2,6}-\d{2,4}$", description="e.g. BOB-001")
    name: str = Field(description="Fake display name")
    bank: str = Field(description="Fake bank name")
    external_account_ref: str = Field(description="Fake external account reference")
    verified: bool = Field(description="Whether the beneficiary passed (simulated) verification")


class Account(BaseModel):
    """A synthetic bank account."""

    id: str = Field(pattern=r"^ACC-SYN-\d+$", description="e.g. ACC-SYN-001")
    owner_name: str = Field(description="Fake owner name")
    balance_inr: int = Field(ge=0, description="Balance in whole INR")
    per_txn_limit_inr: int = Field(gt=0, description="Maximum single transfer amount")
    beneficiary_ids: list[str] = Field(default_factory=list)
    status: AccountStatus = AccountStatus.ACTIVE


class Transaction(BaseModel):
    """One ledger entry (successful or rejected transfer attempt)."""

    id: str = Field(description="e.g. TXN-SYN-0001")
    timestamp: datetime = Field(description="Logical clock value (deterministic, not wall time)")
    tx_type: str = "TRANSFER"
    account_id: str
    beneficiary_id: str
    amount_inr: int
    status: TransactionStatus
    reason_code: str | None = Field(default=None, description="Rejection reason, if REJECTED")
    balance_after_inr: int | None = Field(default=None, description="Balance after a SUCCESS")


class BankState(BaseModel):
    """Complete mutable state of the simulated bank."""

    accounts: dict[str, Account] = Field(default_factory=dict)
    beneficiaries: dict[str, Beneficiary] = Field(default_factory=dict)
    transactions: list[Transaction] = Field(default_factory=list)
    tx_counter: int = Field(default=0, description="Monotonic ledger sequence for deterministic IDs")
    synthetic: bool = Field(
        default=True,
        description="Safety flag: state MUST be synthetic (docs/safety.md)",
    )

    def account_ids(self) -> list[str]:
        return sorted(self.accounts)
