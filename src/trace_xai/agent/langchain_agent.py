"""LLM-driven banking agent built on LangChain AgentExecutor + local Ollama.

Same interface as RuleBasedAgent (run / run_traced -> ExecutionResult), so the
agent is replaceable. Uses a tool-calling agent with qwen3.5:4b (local,
temperature 0). The LLM only *selects tools*; all banking constraints are
enforced by the environment, so LLM nondeterminism cannot bypass rules.

Trace honesty note: through the LangChain tool loop the agent does not observe
environment state directly, so state_before/state_after in LLM traces are
explicitly null (unknown) — never fabricated.

Heavy imports are lazy and USE_TORCH=0 is set defensively: the shared
site-packages on this box contains a CUDA-only torch that breaks `transformers`
imports (see docs/architecture.md).
"""

from __future__ import annotations

import json
import os

from ..banking.environment import BankEnvironment
from ..banking.results import ExecutionResult, ReasonCode, ToolCallRecord
from ..core.recorder import ExecutionRecorder
from ..core.trace import ExecutionTrace
from .parser import parse_task

SYSTEM_PROMPT = (
    "You are a banking assistant operating on a SIMULATED environment with "
    "SYNTHETIC data. Use the provided tools to answer the user's request. "
    "Never invent balances, ids, or outcomes: rely only on tool results. "
    "If a tool reports failure, report its reason_code to the user."
)


class LangChainBankingAgent:
    """Tool-calling agent (LangChain + local Ollama)."""

    name = "langchain-ollama-agent"

    def __init__(
        self,
        environment: BankEnvironment,
        model: str = "qwen3.5:4b",
        max_iterations: int = 4,
    ) -> None:
        # Defensive: keep transformers/torch from breaking langchain imports here.
        os.environ.setdefault("USE_TORCH", "0")
        os.environ.setdefault("USE_TF", "0")

        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_ollama import ChatOllama

        from .langchain_tools import build_tools

        self._env = environment
        tools = build_tools(environment)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        llm = ChatOllama(model=model, temperature=0, seed=42)
        agent = create_tool_calling_agent(llm, tools, prompt)
        self._executor = AgentExecutor(
            agent=agent, tools=tools, max_iterations=max_iterations,
            handle_parsing_errors=True, return_intermediate_steps=True,
        )
        self._task_counter = 0

    def run(self, task: str) -> ExecutionResult:
        result, _trace = self.run_traced(task)
        return result

    def run_traced(self, task: str) -> tuple[ExecutionResult, ExecutionTrace]:
        self._task_counter += 1
        recorder = ExecutionRecorder(
            task_id=f"TASK-LLM-{self._task_counter:04d}",
            task_text=task,
            agent_id=self.name,
            trace_id=f"TRC-LLM-{self._task_counter:04d}",
        )
        recorder.record_task_received()

        raw = self._executor.invoke({"input": task})
        tool_calls: list[ToolCallRecord] = []
        for action, observation in raw.get("intermediate_steps", []):
            result_payload = self._parse_observation(observation)
            args = dict(action.tool_input) if isinstance(action.tool_input, dict) else {}
            invocation = recorder.record_tool_invocation(
                action=action.tool, tool_name=action.tool, tool_input=args,
                state_before=None,  # not observable through the tool loop
            )
            recorder.record_tool_result(
                action=action.tool, tool_name=action.tool, tool_output=result_payload,
                state_after=None, success=bool(result_payload.get("success", False)),
                related_event_id=invocation.event_id,
            )
            tool_calls.append(
                ToolCallRecord(
                    tool=action.tool,
                    args=args,
                    success=bool(result_payload.get("success", False)),
                    result=result_payload,
                    reason_code=ReasonCode(
                        result_payload.get("reason_code", ReasonCode.INTERNAL_ERROR.value)
                    ),
                )
            )

        last = tool_calls[-1] if tool_calls else None
        success = last.success if last else False
        code = last.reason_code if last else ReasonCode.INTERNAL_ERROR
        intent = parse_task(task)

        result = ExecutionResult(
            task=task,
            agent_name=self.name,
            intent_type=intent.type.value,
            action=last.tool if last else "none",
            success=success,
            reason_code=code,
            message=str(raw["output"]),
            tool_calls=tool_calls,
            receipt_id=(last.result.get("transaction_id") if last and success else None),
            llm_used=True,
        )
        recorder.record_outcome(code.value, success=success)
        return result, recorder.finish()

    # ------------------------------------------------------------- internals
    @staticmethod
    def _parse_observation(observation) -> dict:
        try:
            result = json.loads(observation)
            if not isinstance(result, dict):
                result = {"raw": result}
        except (TypeError, ValueError):
            result = {"raw": str(observation)}
        return result
