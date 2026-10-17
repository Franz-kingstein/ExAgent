"""Environment tests: constraint engine, state transitions, determinism."""

import pytest

from trace_xai.banking.dataset import load_dataset
from trace_xai.banking.environment import BankEnvironment
from trace_xai.banking.results import ReasonCode


@pytest.fixture()
def env() -> BankEnvironment:
    return BankEnvironment.from_dataset()


# ---------------------------------------------------------------- balance/history
def test_get_balance(env):
    out = env.get_balance("ACC-SYN-001")
    assert out["success"] is True
    assert out["balance_inr"] == 250000


def test_get_balance_unknown_account(env):
    out = env.get_balance("ACC-SYN-999")
    assert out["success"] is False
    assert out["reason_code"] == ReasonCode.ACCOUNT_NOT_FOUND.value


def test_history_starts_empty(env):
    out = env.get_history("ACC-SYN-001")
    assert out["success"] is True and out["count"] == 0


# ---------------------------------------------------------------- beneficiary
def test_check_beneficiary_verified_and_linked(env):
    out = env.check_beneficiary("BOB-001", "ACC-SYN-001")
    assert out["success"] is True and out["verified"] is True and out["linked"] is True


def test_check_beneficiary_unverified(env):
    out = env.check_beneficiary("BOB-002", "ACC-SYN-001")
    assert out["success"] is True and out["verified"] is False


def test_check_beneficiary_not_linked(env):
    # BOB-003 exists but is not linked to ACC-SYN-002
    out = env.check_beneficiary("BOB-003", "ACC-SYN-002")
    assert out["success"] is False
    assert out["reason_code"] == ReasonCode.BENEFICIARY_NOT_LINKED.value


def test_check_beneficiary_unknown(env):
    out = env.check_beneficiary("BOB-999", "ACC-SYN-001")
    assert out["success"] is False
    assert out["reason_code"] == ReasonCode.BENEFICIARY_NOT_FOUND.value


# ---------------------------------------------------------------- transfer rules
def test_transfer_success_updates_state_and_ledger(env):
    before = env.fingerprint()
    out = env.transfer("BOB-001", 10000, "ACC-SYN-001")
    assert out["success"] is True
    assert out["balance_inr"] == 240000
    assert out["transaction_id"] == "TXN-SYN-0001"
    assert env.fingerprint() != before  # state transitioned
    hist = env.get_history("ACC-SYN-001")
    assert hist["count"] == 1
    tx = hist["transactions"][0]
    assert tx["status"] == "SUCCESS" and tx["balance_after_inr"] == 240000


def test_transfer_exceeding_limit_rejected(env):
    out = env.transfer("BOB-001", 60000, "ACC-SYN-001")
    assert out["success"] is False
    assert out["reason_code"] == ReasonCode.LIMIT_EXCEEDED.value
    assert env.get_balance("ACC-SYN-001")["balance_inr"] == 250000  # funds untouched
    # rejected attempts are recorded in the ledger for auditability
    hist = env.get_history("ACC-SYN-001")
    assert hist["count"] == 1
    assert hist["transactions"][0]["status"] == "REJECTED"


def test_transfer_unverified_beneficiary_rejected(env):
    out = env.transfer("BOB-002", 5000, "ACC-SYN-001")
    assert out["success"] is False
    assert out["reason_code"] == ReasonCode.BENEFICIARY_UNVERIFIED.value
    assert env.get_balance("ACC-SYN-001")["balance_inr"] == 250000


def test_transfer_insufficient_funds_rejected(env):
    out = env.transfer("BOB-001", 10000, "ACC-SYN-002")  # balance is only 5000
    assert out["success"] is False
    assert out["reason_code"] == ReasonCode.INSUFFICIENT_FUNDS.value
    assert env.get_balance("ACC-SYN-002")["balance_inr"] == 5000


def test_transfer_not_linked_and_unknown_beneficiary(env):
    assert env.transfer("BOB-003", 1000, "ACC-SYN-002")["reason_code"] == (
        ReasonCode.BENEFICIARY_NOT_LINKED.value
    )
    assert env.transfer("BOB-999", 1000, "ACC-SYN-001")["reason_code"] == (
        ReasonCode.BENEFICIARY_NOT_FOUND.value
    )


def test_transfer_invalid_amount_rejected(env):
    out = env.transfer("BOB-001", 0, "ACC-SYN-001")
    assert out["success"] is False
    assert out["reason_code"] == ReasonCode.INVALID_AMOUNT.value


def test_sequential_transfers_update_ledger_deterministically(env):
    r1 = env.transfer("BOB-001", 10000, "ACC-SYN-001")
    r2 = env.transfer("BOB-001", 20000, "ACC-SYN-001")
    assert r1["transaction_id"] == "TXN-SYN-0001"
    assert r2["transaction_id"] == "TXN-SYN-0002"
    assert r2["balance_inr"] == 220000
    hist = env.get_history("ACC-SYN-001")
    assert [t["id"] for t in hist["transactions"]] == ["TXN-SYN-0001", "TXN-SYN-0002"]
    # logical clock, not wall time
    assert hist["transactions"][0]["timestamp"] < hist["transactions"][1]["timestamp"]


def test_snapshot_is_isolated_from_state(env):
    snap = env.snapshot()
    snap.accounts["ACC-SYN-001"].balance_inr = 1
    assert env.get_balance("ACC-SYN-001")["balance_inr"] == 250000


def test_fresh_env_same_fingerprint():
    assert BankEnvironment.from_dataset().fingerprint() == (
        BankEnvironment.from_dataset().fingerprint()
    )


def test_default_account_used_when_omitted(env):
    out = env.transfer("BOB-001", 1000, None)
    assert out["success"] is True and out["account_id"] == "ACC-SYN-001"
