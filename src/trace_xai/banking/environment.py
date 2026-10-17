"""Simulated banking environment: tools + transfer constraint engine.

SYNTHETIC DATA ONLY (docs/safety.md). Nothing here talks to any network,
payment system, or real financial API. All timestamps come from a logical
clock so behavior is fully deterministic.

The environment is replaceable: consumers depend on BankEnvironmentProtocol,
not on this class.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from .dataset import DEFAULT_DATASET_PATH, load_dataset
from .results import ReasonCode
from .state import Account, AccountStatus, BankState, Transaction, TransactionStatus


@runtime_checkable
class BankEnvironmentProtocol(Protocol):
    """Interface every banking environment must implement (replaceability)."""

    def get_balance(self, account_id: str | None = None) -> dict: ...

    def check_beneficiary(self, beneficiary_id: str, account_id: str | None = None) -> dict: ...

    def get_history(self, account_id: str | None = None, limit: int | None = None) -> dict: ...

    def transfer(
        self, beneficiary_id: str, amount_inr: int, account_id: str | None = None
    ) -> dict: ...


class BankEnvironment:
    """Deterministic simulated bank built on BankState."""

    def __init__(
        self,
        state: BankState,
        default_account_id: str | None = None,
        clock_start: datetime | None = None,
    ) -> None:
        self._state = state
        ids = state.account_ids()
        self._default_account_id = default_account_id or (ids[0] if ids else None)
        self._clock_start = clock_start or datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    @classmethod
    def from_dataset(cls, path: str | None = None, **kwargs) -> "BankEnvironment":
        return cls(load_dataset(path or DEFAULT_DATASET_PATH), **kwargs)

    # ------------------------------------------------------------------ state
    @property
    def default_account_id(self) -> str | None:
        return self._default_account_id

    def snapshot(self) -> BankState:
        """Deep copy of current state (safe for inspection/tests)."""
        return self._state.model_copy(deep=True)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the state (tamper/transition checks)."""
        payload = json.dumps(self._state.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        """Compact, deterministic observable-state summary for traces.

        This is what trace events record in state_before/state_after; it is
        an *observation*, not a copy of the full state. XAI matchers rely on
        these fields as evidence, so they must stay honest and complete.
        """
        return {
            "fingerprint": self.fingerprint(),
            "balances": {
                aid: acc.balance_inr for aid, acc in sorted(self._state.accounts.items())
            },
            "per_txn_limits": {
                aid: acc.per_txn_limit_inr for aid, acc in sorted(self._state.accounts.items())
            },
            "beneficiaries_verified": {
                bid: b.verified for bid, b in sorted(self._state.beneficiaries.items())
            },
            "ledger_size": len(self._state.transactions),
            "tx_counter": self._state.tx_counter,
        }

    # ------------------------------------------------------------------ tools
    def get_balance(self, account_id: str | None = None) -> dict:
        account_id = account_id or self._default_account_id
        account = self._state.accounts.get(account_id)
        if account is None:
            return self._tool_result(False, ReasonCode.ACCOUNT_NOT_FOUND, account_id=account_id)
        return self._tool_result(
            True, ReasonCode.OK, account_id=account.id, balance_inr=account.balance_inr
        )

    def check_beneficiary(self, beneficiary_id: str, account_id: str | None = None) -> dict:
        account_id = account_id or self._default_account_id
        account = self._require_account(account_id)
        if account is None:
            return self._tool_result(
                False, ReasonCode.ACCOUNT_NOT_FOUND, account_id=account_id,
                beneficiary_id=beneficiary_id,
            )
        beneficiary = self._state.beneficiaries.get(beneficiary_id)
        if beneficiary is None:
            return self._tool_result(
                False, ReasonCode.BENEFICIARY_NOT_FOUND,
                account_id=account.id, beneficiary_id=beneficiary_id,
            )
        linked = beneficiary_id in account.beneficiary_ids
        return self._tool_result(
            linked, ReasonCode.OK if linked else ReasonCode.BENEFICIARY_NOT_LINKED,
            account_id=account.id, beneficiary_id=beneficiary.id, name=beneficiary.name,
            bank=beneficiary.bank, verified=beneficiary.verified, linked=linked,
        )

    def get_history(self, account_id: str | None = None, limit: int | None = None) -> dict:
        account_id = account_id or self._default_account_id
        account = self._require_account(account_id)
        if account is None:
            return self._tool_result(False, ReasonCode.ACCOUNT_NOT_FOUND, account_id=account_id)
        entries = [t for t in self._state.transactions if t.account_id == account.id]
        if limit is not None:
            entries = entries[-limit:]
        return self._tool_result(
            True, ReasonCode.OK, account_id=account.id, count=len(entries),
            transactions=[t.model_dump(mode="json") for t in entries],
        )

    def transfer(self, beneficiary_id: str, amount_inr: int, account_id: str | None = None) -> dict:
        """Execute a transfer subject to banking constraints.

        Rejected attempts (except unknown accounts) are recorded in the
        ledger with status=REJECTED for auditability.
        """
        account_id = account_id or self._default_account_id
        account = self._require_account(account_id)
        if account is None:
            return self._tool_result(
                False, ReasonCode.ACCOUNT_NOT_FOUND, account_id=account_id,
                beneficiary_id=beneficiary_id, amount_inr=amount_inr,
            )

        reason = self._check_transfer(account, beneficiary_id, amount_inr)
        if reason is not None:
            tx = self._record_transaction(account, beneficiary_id, amount_inr, reason)
            return self._tool_result(
                False, reason, account_id=account.id, beneficiary_id=beneficiary_id,
                amount_inr=amount_inr, balance_inr=account.balance_inr,
                transaction_id=tx.id,
            )

        account.balance_inr -= amount_inr
        tx = self._record_transaction(
            account, beneficiary_id, amount_inr, None, balance_after=account.balance_inr
        )
        return self._tool_result(
            True, ReasonCode.OK, account_id=account.id, beneficiary_id=beneficiary_id,
            amount_inr=amount_inr, balance_inr=account.balance_inr, transaction_id=tx.id,
        )

    # ------------------------------------------------------------- internals
    def _require_account(self, account_id: str | None) -> Account | None:
        if account_id is None:
            return None
        return self._state.accounts.get(account_id)

    def _check_transfer(
        self, account: Account, beneficiary_id: str, amount_inr: int
    ) -> ReasonCode | None:
        """Return the first violated constraint, or None if the transfer is valid."""
        if account.status != AccountStatus.ACTIVE:
            return ReasonCode.ACCOUNT_NOT_ACTIVE
        if not isinstance(amount_inr, int) or amount_inr <= 0:
            return ReasonCode.INVALID_AMOUNT
        beneficiary = self._state.beneficiaries.get(beneficiary_id)
        if beneficiary is None:
            return ReasonCode.BENEFICIARY_NOT_FOUND
        if beneficiary_id not in account.beneficiary_ids:
            return ReasonCode.BENEFICIARY_NOT_LINKED
        if not beneficiary.verified:
            return ReasonCode.BENEFICIARY_UNVERIFIED
        if amount_inr > account.per_txn_limit_inr:
            return ReasonCode.LIMIT_EXCEEDED
        if amount_inr > account.balance_inr:
            return ReasonCode.INSUFFICIENT_FUNDS
        return None

    def _record_transaction(
        self,
        account: Account,
        beneficiary_id: str,
        amount_inr: int,
        reason: ReasonCode | None,
        balance_after: int | None = None,
    ) -> Transaction:
        self._state.tx_counter += 1
        tx = Transaction(
            id=f"TXN-SYN-{self._state.tx_counter:04d}",
            timestamp=self._clock_start + timedelta(minutes=self._state.tx_counter),
            account_id=account.id,
            beneficiary_id=beneficiary_id,
            amount_inr=amount_inr,
            status=TransactionStatus.SUCCESS if reason is None else TransactionStatus.REJECTED,
            reason_code=None if reason is None else reason.value,
            balance_after_inr=balance_after,
        )
        self._state.transactions.append(tx)
        return tx

    @staticmethod
    def _tool_result(success: bool, reason: ReasonCode, **payload) -> dict:
        """Uniform tool result shape used by every tool (evidence-friendly)."""
        return {"success": success, "reason_code": reason.value, **payload}
