"""Agent package: replaceable banking agents.

- RuleBasedAgent: deterministic, default for tests and reproducible runs.
- LangChainBankingAgent: LLM tool-calling agent (local Ollama), optional.
"""

from __future__ import annotations

from ..banking.environment import BankEnvironment
from .base import BankingAgentProtocol
from .parser import Intent, IntentType, parse_task
from .rule_agent import RuleBasedAgent


def make_agent(
    kind: str, environment: BankEnvironment, model: str = "qwen3.5:4b"
) -> BankingAgentProtocol:
    """Factory: kind='rule' (deterministic) or kind='llm' (LangChain + Ollama)."""
    if kind == "rule":
        return RuleBasedAgent(environment)
    if kind == "llm":
        from .langchain_agent import LangChainBankingAgent

        return LangChainBankingAgent(environment, model=model)
    raise ValueError(f"Unknown agent kind: {kind!r} (expected 'rule' or 'llm')")


__all__ = [
    "BankingAgentProtocol",
    "RuleBasedAgent",
    "Intent",
    "IntentType",
    "parse_task",
    "make_agent",
]
