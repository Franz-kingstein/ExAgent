"""LangChain tool bindings over the banking environment.

Adapter layer only: it exposes BankEnvironmentProtocol tools to a LangChain
AgentExecutor. No business logic lives here — every rule stays in the
environment so agents (rule-based or LLM) face identical constraints.

Tool results are returned as JSON strings with the uniform shape
{"success": bool, "reason_code": str, ...} so they can be parsed back into
ToolCallRecord entries as evidence.
"""

from __future__ import annotations

import json

from ..banking.environment import BankEnvironment


def build_tools(env: BankEnvironment) -> list:
    """Build LangChain StructuredTools bound to one environment instance."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class BalanceArgs(BaseModel):
        account_id: str | None = Field(
            default=None, description="Synthetic account id, e.g. ACC-SYN-001"
        )

    class BeneficiaryArgs(BaseModel):
        beneficiary_id: str = Field(..., description="Beneficiary id, e.g. BOB-001")
        account_id: str | None = Field(default=None, description="Account id, e.g. ACC-SYN-001")

    class HistoryArgs(BaseModel):
        account_id: str | None = Field(default=None, description="Account id, e.g. ACC-SYN-001")
        limit: int | None = Field(default=None, description="Return only the last N entries")

    class TransferArgs(BaseModel):
        beneficiary_id: str = Field(..., description="Verified beneficiary id, e.g. BOB-001")
        amount_inr: int = Field(..., description="Positive integer amount in INR")
        account_id: str | None = Field(default=None, description="Debit account id")

    def get_balance(account_id: str | None = None) -> str:
        """Check the balance of a synthetic bank account."""
        return json.dumps(env.get_balance(account_id))

    def check_beneficiary(beneficiary_id: str, account_id: str | None = None) -> str:
        """Check whether a beneficiary exists, is linked to the account, and is verified."""
        return json.dumps(env.check_beneficiary(beneficiary_id, account_id))

    def get_history(account_id: str | None = None, limit: int | None = None) -> str:
        """List transaction history (successful and rejected) for a synthetic account."""
        return json.dumps(env.get_history(account_id, limit))

    def transfer(beneficiary_id: str, amount_inr: int, account_id: str | None = None) -> str:
        """Transfer money to a beneficiary. Fails if unverified, over limit, or insufficient funds."""
        return json.dumps(env.transfer(beneficiary_id, amount_inr, account_id))

    return [
        StructuredTool.from_function(get_balance, name="get_balance", args_schema=BalanceArgs),
        StructuredTool.from_function(
            check_beneficiary, name="check_beneficiary", args_schema=BeneficiaryArgs
        ),
        StructuredTool.from_function(get_history, name="get_history", args_schema=HistoryArgs),
        StructuredTool.from_function(transfer, name="transfer", args_schema=TransferArgs),
    ]
