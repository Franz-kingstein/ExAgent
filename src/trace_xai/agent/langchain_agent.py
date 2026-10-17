"""LLM-driven banking agent built on LangChain AgentExecutor + local Ollama.

Same interface as RuleBasedAgent (run(task) -> ExecutionResult), so the agent
is replaceable. Uses a tool-calling agent with qwen3.5:4b (local, temperature
0). The LLM only *selects tools*; all banking constraints are enforced by the
environment, so LLM nondeterminism cannot bypass rules.

Heavy imports are lazy and USE_TORCH=0 is set defensively: the shared
site-packages on this box contains a CUDA-only torch that breaks `transformers`
imports (see docs/architecture.md).
"""

from __future__ import annotations

import json
import os

from ..banking.environment import BankEnvironment
from ..banking.results import ExecutionResult, ReasonCode, ToolCallRecord
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

    def run(self, task: str) -> ExecutionResult:
        raw = self._executor.invoke({"input": task})
        tool_calls = [
            self._to_record(action, observation)
            for action, observation in raw.get("intermediate_steps", [])
        ]
        intent = parse_task(task)

        last = tool_calls[-1] if tool_calls else None
        success = last.success if last else False
        code = last.reason_code if last else ReasonCode.INTERNAL_ERROR

        return ExecutionResult(
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

    @staticmethod
    def _to_record(action, observation: str) -> ToolCallRecord:
        try:
            result = json.loads(observation)
            if not isinstance(result, dict):
                result = {"raw": result}
        except (TypeError, ValueError):
            result = {"raw": str(observation)}
        return ToolCallRecord(
            tool=action.tool,
            args=dict(action.tool_input) if isinstance(action.tool_input, dict) else {},
            success=bool(result.get("success", False)),
            result=result,
            reason_code=ReasonCode(result.get("reason_code", ReasonCode.INTERNAL_ERROR.value)),
        )
