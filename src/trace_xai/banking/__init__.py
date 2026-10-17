"""Simulated banking domain: state, dataset, environment, results.

Everything here operates on SYNTHETIC data only (see docs/safety.md).
"""

from .state import Account, BankState, Beneficiary, Transaction
from .dataset import load_dataset
from .environment import BankEnvironment, BankEnvironmentProtocol
from .results import ExecutionResult, ReasonCode, ToolCallRecord

__all__ = [
    "Account",
    "BankState",
    "Beneficiary",
    "Transaction",
    "load_dataset",
    "BankEnvironment",
    "BankEnvironmentProtocol",
    "ExecutionResult",
    "ReasonCode",
    "ToolCallRecord",
]
