"""Machine-readable requirement registry (deterministic constraint types).

Requirements are loaded from configs/requirements_bank.yaml. Constraint types
are a closed set evaluated by structured comparison — no LLM involvement.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_REQUIREMENTS_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "requirements_bank.yaml"
)


class RequirementsError(ValueError):
    """Raised when the requirement registry is invalid."""


class ConstraintType(str, Enum):
    """Closed set of deterministically evaluable constraints."""

    AMOUNT_WITHIN_LIMIT = "amount_within_limit"
    BENEFICIARY_VERIFIED = "beneficiary_verified"
    SUFFICIENT_BALANCE = "sufficient_balance"


class Constraint(BaseModel):
    type: ConstraintType


class Requirement(BaseModel):
    id: str = Field(pattern=r"^REQ-[A-Z]+-\d+$", description="e.g. REQ-AMT-001")
    text: str
    applies_to: str = Field(description="Action name (e.g. TRANSFER) or ANY")
    constraint: Constraint


class RequirementSet(BaseModel):
    requirements: list[Requirement]
    synthetic: bool = True
    version: int = 1

    def applicable_to(self, action: str | None) -> list[Requirement]:
        if action is None:
            return []
        action = action.upper()
        return [r for r in self.requirements if r.applies_to.upper() in (action, "ANY")]

    def by_id(self, requirement_id: str) -> Requirement | None:
        return next((r for r in self.requirements if r.id == requirement_id), None)


def load_requirements(path: str | Path | None = None) -> RequirementSet:
    """Load and validate the requirement registry (synthetic-only gate)."""
    path = Path(path) if path is not None else DEFAULT_REQUIREMENTS_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("meta", {}).get("synthetic") is not True:
        raise RequirementsError(
            "Refusing to load a requirement registry not marked meta.synthetic=true "
            "(docs/safety.md: synthetic data only)."
        )
    reqs = RequirementSet(
        requirements=[Requirement.model_validate(r) for r in raw.get("requirements", [])],
        synthetic=True,
        version=int(raw["meta"].get("version", 1)),
    )
    ids = [r.id for r in reqs.requirements]
    if len(set(ids)) != len(ids):
        raise RequirementsError("Duplicate requirement ids")
    if not reqs.requirements:
        raise RequirementsError("Requirement registry is empty")
    return reqs
