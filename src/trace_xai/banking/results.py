"""Execution result models: the structured outcome of one agent run.

This is the *result* layer, kept separate from agent, environment, and state
so that any agent (rule-based or LLM) and any environment can emit the same
record shape. Later TRACE-XAI phases will consume these records as evidence.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ReasonCode(str, Enum):
    """Uniform reason codes shared by environment tools and agents."""

    OK = "OK"
    ACCOUNT_NOT_FOUND = "ACCOUNT_NOT_FOUND"
    ACCOUNT_NOT_ACTIVE = "ACCOUNT_NOT_ACTIVE"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    BENEFICIARY_NOT_FOUND = "BENEFICIARY_NOT_FOUND"
    BENEFICIARY_NOT_LINKED = "BENEFICIARY_NOT_LINKED"
    BENEFICIARY_UNVERIFIED = "BENEFICIARY_UNVERIFIED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    UNPARSEABLE_TASK = "UNPARSEABLE_TASK"
    TASK_NOT_SUPPORTED = "TASK_NOT_SUPPORTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolCallRecord(BaseModel):
    """One observed tool invocation (trace element for later explanation)."""

    tool: str
    args: dict = Field(default_factory=dict)
    success: bool
    result: dict = Field(default_factory=dict)
    reason_code: ReasonCode = ReasonCode.INTERNAL_ERROR


class ExecutionResult(BaseModel):
    """Full, structured outcome of an agent run."""

    task: str = Field(description="Raw natural-language task")
    agent_name: str
    intent_type: str = Field(description="Parsed intent type, or UNKNOWN")
    action: str = Field(description="Final action taken, e.g. 'transfer'")
    success: bool
    reason_code: ReasonCode
    message: str = Field(description="Human-readable summary")
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    receipt_id: str | None = Field(default=None, description="Ledger id for committed transfers")
    llm_used: bool = Field(default=False)
