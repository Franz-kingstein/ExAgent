"""Agent-trace integration tests (deterministic rule-based agent).

Covers the Phase 2 requirements: complete traces on success and failure,
captured tool inputs/outputs, captured state transitions, explicitly-missing
evidence, ordering, and serialization — including the committed sample trace
usable without re-running the agent.
"""

from pathlib import Path

import pytest

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.trace import EventType, ExecutionTrace

SAMPLE_TRACE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "traces" / "sample_transfer_success.json"
)


@pytest.fixture()
def env() -> BankEnvironment:
    return BankEnvironment.from_dataset()


@pytest.fixture()
def agent(env) -> RuleBasedAgent:
    return RuleBasedAgent(env)


def test_successful_execution_generates_complete_trace(agent):
    _result, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
    assert [e.event_type for e in trace.events] == [
        EventType.TASK_RECEIVED,
        EventType.TOOL_INVOCATION,
        EventType.TOOL_RESULT,
        EventType.OUTCOME_RECORDED,
    ]
    success, outcome = trace.final_outcome
    assert success is True and outcome == "OK"
    assert trace.task_text.startswith("Transfer")
    assert trace.agent_id == "rule-based-agent"


def test_failed_execution_generates_trace(agent):
    _result, trace = agent.run_traced("Transfer ₹60,000 to beneficiary BOB-001.")
    assert len(trace.events) == 4  # same structure as success
    success, outcome = trace.final_outcome
    assert success is False and outcome == "LIMIT_EXCEEDED"


def test_tool_inputs_and_outputs_are_captured(agent):
    _result, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
    invocation, tool_result = trace.events[1], trace.events[2]
    assert invocation.tool_name == "transfer"
    assert invocation.tool_input == {"beneficiary_id": "BOB-001", "amount_inr": 10000}
    assert tool_result.tool_output["success"] is True
    assert tool_result.tool_output["transaction_id"] == "TXN-SYN-0001"
    assert tool_result.tool_output["balance_inr"] == 240000
    assert tool_result.related_event_id == invocation.event_id


def test_state_changes_are_captured(agent, env):
    _result, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
    invocation, tool_result = trace.events[1], trace.events[2]
    assert invocation.state_before["balances"]["ACC-SYN-001"] == 250000
    assert tool_result.state_after["balances"]["ACC-SYN-001"] == 240000
    assert tool_result.state_after["fingerprint"] != invocation.state_before["fingerprint"]


def test_rejected_transfer_state_shows_no_money_moved(agent):
    _result, trace = agent.run_traced("Transfer ₹60,000 to beneficiary BOB-001.")
    before, after = trace.events[1].state_before, trace.events[2].state_after
    assert after["balances"] == before["balances"]  # no funds moved
    assert after["ledger_size"] == before["ledger_size"] + 1  # rejection recorded


def test_missing_evidence_stays_explicitly_missing(agent):
    _result, trace = agent.run_traced("Verify beneficiary BOB-001")
    for event in trace.events:
        # no requirement engine yet: references/expectations must be null,
        # not empty strings or placeholders
        assert event.requirement_reference is None
        assert event.expected_outcome is None
    assert trace.events[1].tool_input == {"beneficiary_id": "BOB-001"}
    assert trace.events[1].state_before is not None  # rule agent does observe state
    assert trace.events[2].tool_output["verified"] is True
    assert trace.events[2].success is True


def test_unparseable_task_traces_failure_without_tools(agent):
    _result, trace = agent.run_traced("Order a pizza with extra cheese")
    assert [e.event_type for e in trace.events] == [
        EventType.TASK_RECEIVED, EventType.OUTCOME_RECORDED,
    ]
    assert trace.final_outcome == (False, "UNPARSEABLE_TASK")


def test_trace_entries_maintain_correct_ordering(agent):
    _result, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
    seqs = [e.sequence for e in trace.events]
    assert seqs == sorted(seqs) == [1, 2, 3, 4]
    stamps = [e.timestamp for e in trace.events]
    assert stamps == sorted(stamps)


def test_trace_serialization_round_trip(agent):
    _result, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
    assert ExecutionTrace.from_json(trace.to_json()) == trace


def test_traces_are_deterministic():
    def run_once() -> str:
        agent = RuleBasedAgent(BankEnvironment.from_dataset())
        return agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")[1].to_json()

    assert run_once() == run_once()


def test_committed_sample_trace_is_valid_and_standalone():
    """data/traces/sample_transfer_success.json must load without any agent run."""
    trace = ExecutionTrace.from_json(SAMPLE_TRACE_PATH.read_text(encoding="utf-8"))
    assert trace.agent_id == "rule-based-agent"
    assert len(trace.events) == 4
    assert trace.final_outcome == (True, "OK")
    assert trace.events[2].tool_output["transaction_id"] == "TXN-SYN-0001"
    assert all(e.provenance.synthetic for e in trace.events)
