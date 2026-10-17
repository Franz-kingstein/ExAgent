"""Agent protocol: any banking agent must implement run(task) -> ExecutionResult.

Agents are replaceable: components consume this protocol, never concrete
implementations (rule-based today, LLM-driven later).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..banking.results import ExecutionResult


@runtime_checkable
class BankingAgentProtocol(Protocol):
    name: str

    def run(self, task: str) -> ExecutionResult: ...
