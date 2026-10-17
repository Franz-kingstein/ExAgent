"""Dataset loader tests: determinism, structure, and the synthetic-only gate."""

import pytest

from trace_xai.banking.dataset import DatasetError, load_dataset
from trace_xai.banking.state import AccountStatus


def test_dataset_loads_and_is_synthetic():
    state = load_dataset()
    assert state.synthetic is True
    assert set(state.accounts) == {"ACC-SYN-001", "ACC-SYN-002"}
    assert set(state.beneficiaries) == {"BOB-001", "BOB-002", "BOB-003"}


def test_dataset_is_deterministic():
    a, b = load_dataset(), load_dataset()
    assert a.model_dump() == b.model_dump()


def test_synthetic_id_conventions():
    state = load_dataset()
    assert all(acc.id.startswith("ACC-SYN-") for acc in state.accounts.values())
    assert all(bob.id.startswith("BOB-") for bob in state.beneficiaries.values())
    assert all(
        ref.external_account_ref.startswith("SYN-EXT-")
        for ref in state.beneficiaries.values()
    )


def test_expected_fixture_values():
    state = load_dataset()
    acc1, acc2 = state.accounts["ACC-SYN-001"], state.accounts["ACC-SYN-002"]
    assert acc1.balance_inr == 250000 and acc1.per_txn_limit_inr == 50000
    assert acc2.balance_inr == 5000 and acc2.per_txn_limit_inr == 10000
    assert acc1.status is AccountStatus.ACTIVE
    assert state.beneficiaries["BOB-001"].verified is True
    assert state.beneficiaries["BOB-002"].verified is False


def test_linked_beneficiaries_exist():
    state = load_dataset()
    for acc in state.accounts.values():
        for bid in acc.beneficiary_ids:
            assert bid in state.beneficiaries


def test_dataset_rejects_non_synthetic(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "meta: {synthetic: false}\naccounts: []\nbeneficiaries: []\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError):
        load_dataset(bad)
