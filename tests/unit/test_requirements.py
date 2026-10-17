"""Requirement registry loader tests."""

import pytest

from trace_xai.requirements import (
    ConstraintType,
    RequirementSet,
    RequirementsError,
    load_requirements,
)


def test_registry_loads():
    reqs = load_requirements()
    assert reqs.synthetic is True
    assert {r.id for r in reqs.requirements} == {"REQ-AMT-001", "REQ-BEN-001", "REQ-BAL-001"}
    assert reqs.by_id("REQ-AMT-001").constraint.type is ConstraintType.AMOUNT_WITHIN_LIMIT


def test_applicable_filtering():
    reqs = load_requirements()
    assert len(reqs.applicable_to("TRANSFER")) == 3
    assert reqs.applicable_to("BALANCE") == []
    assert reqs.applicable_to(None) == []
    assert len(reqs.applicable_to("transfer")) == 3  # case-insensitive


def test_registry_rejects_non_synthetic(tmp_path):
    bad = tmp_path / "req.yaml"
    bad.write_text(
        "meta: {synthetic: false, version: 1}\nrequirements: []\n", encoding="utf-8"
    )
    with pytest.raises(RequirementsError):
        load_requirements(bad)


def test_registry_rejects_duplicate_ids(tmp_path):
    bad = tmp_path / "req.yaml"
    bad.write_text(
        "meta: {synthetic: true, version: 1}\n"
        "requirements:\n"
        "  - {id: REQ-X-001, text: t, applies_to: TRANSFER, constraint: {type: amount_within_limit}}\n"
        "  - {id: REQ-X-001, text: t, applies_to: TRANSFER, constraint: {type: beneficiary_verified}}\n",
        encoding="utf-8",
    )
    with pytest.raises(RequirementsError, match="Duplicate"):
        load_requirements(bad)


def test_registry_rejects_unknown_constraint(tmp_path):
    bad = tmp_path / "req.yaml"
    bad.write_text(
        "meta: {synthetic: true, version: 1}\n"
        "requirements:\n"
        "  - {id: REQ-X-001, text: t, applies_to: TRANSFER, constraint: {type: mind_reading}}\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_requirements(bad)
