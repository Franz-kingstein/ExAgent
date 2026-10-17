"""Deterministic synthetic dataset loader (SYNTHETIC DATA ONLY).

The dataset is a small YAML file (configs/bank_dataset.yaml). It is fully
deterministic: no randomness, no wall-clock, no external I/O.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .state import Account, AccountStatus, BankState, Beneficiary

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[3] / "configs" / "bank_dataset.yaml"


class DatasetError(ValueError):
    """Raised when the synthetic dataset is invalid."""


def _validate_raw(raw: dict) -> None:
    if not isinstance(raw, dict):
        raise DatasetError("Dataset root must be a mapping")
    if raw.get("meta", {}).get("synthetic") is not True:
        raise DatasetError(
            "Refusing to load a dataset not marked meta.synthetic=true "
            "(docs/safety.md: synthetic data only)."
        )

    accounts = raw.get("accounts") or []
    beneficiaries = raw.get("beneficiaries") or []
    if not accounts:
        raise DatasetError("Dataset must define at least one account")
    if not beneficiaries:
        raise DatasetError("Dataset must define at least one beneficiary")

    account_ids = [a.get("id") for a in accounts]
    beneficiary_ids = [b.get("id") for b in beneficiaries]
    if len(set(account_ids)) != len(account_ids):
        raise DatasetError("Duplicate account ids")
    if len(set(beneficiary_ids)) != len(beneficiary_ids):
        raise DatasetError("Duplicate beneficiary ids")

    known_b = set(beneficiary_ids)
    for a in accounts:
        missing = [b for b in a.get("beneficiary_ids", []) if b not in known_b]
        if missing:
            raise DatasetError(f"Account {a.get('id')} links unknown beneficiaries: {missing}")


def load_dataset(path: str | Path | None = None) -> BankState:
    """Load and validate the synthetic dataset into a BankState."""
    path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_raw(raw)

    accounts = {
        a["id"]: Account(
            id=a["id"],
            owner_name=a["owner_name"],
            balance_inr=int(a["balance_inr"]),
            per_txn_limit_inr=int(a["per_txn_limit_inr"]),
            beneficiary_ids=list(a.get("beneficiary_ids", [])),
            status=AccountStatus(a.get("status", "ACTIVE")),
        )
        for a in raw["accounts"]
    }
    beneficiaries = {
        b["id"]: Beneficiary(
            id=b["id"],
            name=b["name"],
            bank=b["bank"],
            external_account_ref=b["external_account_ref"],
            verified=bool(b["verified"]),
        )
        for b in raw["beneficiaries"]
    }
    return BankState(accounts=accounts, beneficiaries=beneficiaries)
