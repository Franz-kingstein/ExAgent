"""Deterministic rule-based banking agent (default agent for tests).

Flow: parse task -> typed Intent -> dispatch to one environment tool ->
ExecutionResult. No LLM, no randomness: given the same task and environment
state, output is byte-identical.

Every execution also produces a structured ExecutionTrace (Phase 2): only
observed steps are recorded; requirement references and expected outcomes
stay null until a requirements engine exists (Phase 3) — never fabricated.
"""

from __future__ import annotations

from ..banking.environment import BankEnvironment
from ..banking.results import ExecutionResult, ReasonCode, ToolCallRecord
from ..core.recorder import ExecutionRecorder
from ..core.trace import ExecutionTrace
from .parser import Intent, IntentType, parse_task

_MESSAGES = {
    ReasonCode.OK: "Completed successfully.",
    ReasonCode.ACCOUNT_NOT_FOUND: "Account not found.",
    ReasonCode.INVALID_AMOUNT: "Amount must be a positive integer in INR.",
    ReasonCode.BENEFICIARY_NOT_FOUND: "Beneficiary does not exist.",
    ReasonCode.BENEFICIARY_NOT_LINKED: "Beneficiary is not linked to this account.",
    ReasonCode.BENEFICIARY_UNVERIFIED: "Beneficiary is not verified.",
    ReasonCode.LIMIT_EXCEEDED: "Amount exceeds the per-transaction limit.",
    ReasonCode.INSUFFICIENT_FUNDS: "Insufficient balance.",
    ReasonCode.UNPARSEABLE_TASK: "Could not parse the task into a supported intent.",
}

_TOOL_BY_INTENT = {
    IntentType.BALANCE: "get_balance",
    IntentType.BENEFICIARY_CHECK: "check_beneficiary",
    IntentType.HISTORY: "get_history",
    IntentType.TRANSFER: "transfer",
}


class RuleBasedAgent:
    """Deterministic agent: no LLM, fully reproducible."""

    name = "rule-based-agent"

    def __init__(self, environment: BankEnvironment) -> None:
        self._env = environment
        self._task_counter = 0

    def run(self, task: str) -> ExecutionResult:
        """Execute a task and return only the result (trace discarded)."""
        result, _trace = self.run_traced(task)
        return result

    def run_traced(self, task: str) -> tuple[ExecutionResult, ExecutionTrace]:
        """Execute a task and return (result, structured trace)."""
        self._task_counter += 1
        recorder = ExecutionRecorder(
            task_id=f"TASK-SYN-{self._task_counter:04d}",
            task_text=task,
            agent_id=self.name,
            trace_id=f"TRC-SYN-{self._task_counter:04d}",
        )
        recorder.record_task_received()
        intent = parse_task(task)

        if intent.type is IntentType.UNKNOWN or (
            intent.type is IntentType.TRANSFER and intent.amount_inr is None
        ):
            result = self._failure(
                task, intent, ReasonCode.UNPARSEABLE_TASK,
                intent.note or _MESSAGES[ReasonCode.UNPARSEABLE_TASK],
            )
            recorder.record_outcome(result.reason_code.value, success=False)
            return result, recorder.finish()
        if intent.type is IntentType.BENEFICIARY_CHECK and intent.beneficiary_id is None:
            result = self._failure(
                task, intent, ReasonCode.UNPARSEABLE_TASK, "No beneficiary id in task."
            )
            recorder.record_outcome(result.reason_code.value, success=False)
            return result, recorder.finish()

        tool_name = _TOOL_BY_INTENT[intent.type]
        action = intent.type.value
        args = self._args_for(intent)

        invocation = recorder.record_tool_invocation(
            action=action, tool_name=tool_name, tool_input=args,
            state_before=self._env.summary(),
        )
        result_dict = self._dispatch(intent)
        recorder.record_tool_result(
            action=action, tool_name=tool_name, tool_output=result_dict,
            state_after=self._env.summary(),
            success=bool(result_dict["success"]),
            related_event_id=invocation.event_id,
        )

        record = ToolCallRecord(
            tool=tool_name,
            args=args,
            success=bool(result_dict["success"]),
            result=result_dict,
            reason_code=ReasonCode(result_dict["reason_code"]),
        )
        result = self._build_result(task, intent, record, result_dict)
        recorder.record_outcome(result.reason_code.value, success=result.success)
        return result, recorder.finish()

    # ------------------------------------------------------------- internals
    def _dispatch(self, intent: Intent) -> dict:
        if intent.type is IntentType.BALANCE:
            return self._env.get_balance(intent.account_id)
        if intent.type is IntentType.BENEFICIARY_CHECK:
            return self._env.check_beneficiary(intent.beneficiary_id, intent.account_id)
        if intent.type is IntentType.HISTORY:
            return self._env.get_history(intent.account_id)
        if intent.type is IntentType.TRANSFER:
            return self._env.transfer(intent.beneficiary_id, intent.amount_inr, intent.account_id)
        raise ValueError(f"Unhandled intent: {intent.type}")

    @staticmethod
    def _args_for(intent: Intent) -> dict:
        args: dict = {}
        if intent.account_id:
            args["account_id"] = intent.account_id
        if intent.beneficiary_id:
            args["beneficiary_id"] = intent.beneficiary_id
        if intent.amount_inr is not None:
            args["amount_inr"] = intent.amount_inr
        return args

    @staticmethod
    def _build_result(
        task: str, intent: Intent, record: ToolCallRecord, result_dict: dict
    ) -> ExecutionResult:
        message = _MESSAGES[record.reason_code]
        if intent.type is IntentType.BALANCE and record.success:
            message = f"Balance of {result_dict['account_id']}: INR {result_dict['balance_inr']}"
        elif intent.type is IntentType.TRANSFER and record.success:
            message = (
                f"Transferred INR {result_dict['amount_inr']} to {result_dict['beneficiary_id']}; "
                f"new balance INR {result_dict['balance_inr']} "
                f"(ref {result_dict['transaction_id']})"
            )
        return ExecutionResult(
            task=task,
            agent_name="rule-based-agent",
            intent_type=intent.type.value,
            action=record.tool,
            success=record.success,
            reason_code=record.reason_code,
            message=message,
            tool_calls=[record],
            receipt_id=result_dict.get("transaction_id") if record.success else None,
            llm_used=False,
        )

    @staticmethod
    def _failure(task: str, intent: Intent, code: ReasonCode, message: str) -> ExecutionResult:
        return ExecutionResult(
            task=task,
            agent_name="rule-based-agent",
            intent_type=intent.type.value,
            action="none",
            success=False,
            reason_code=code,
            message=message,
            tool_calls=[],
            llm_used=False,
        )
