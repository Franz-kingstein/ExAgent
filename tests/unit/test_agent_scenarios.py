"""End-to-end scenarios A-F with the deterministic rule-based agent.

Each scenario has a fixed expected outcome (reason codes, balances, ledger).
"""

import pytest

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.dataset import load_dataset
from trace_xai.banking.environment import BankEnvironment
from trace_xai.banking.results import ReasonCode


@pytest.fixture()
def env() -> BankEnvironment:
    return BankEnvironment.from_dataset()


@pytest.fixture()
def agent(env) -> RuleBasedAgent:
    return RuleBasedAgent(env)


def test_scenario_a_successful_balance_check(agent):
    result = agent.run("What is the balance of account ACC-SYN-001?")
    assert result.success is True
    assert result.reason_code is ReasonCode.OK
    assert result.action == "get_balance"
    assert "250000" in result.message
    assert result.llm_used is False


def test_scenario_b_successful_beneficiary_verification(agent):
    result = agent.run("Verify beneficiary BOB-001")
    assert result.success is True
    assert result.reason_code is ReasonCode.OK
    assert result.action == "check_beneficiary"
    assert result.tool_calls[0].result["verified"] is True


def test_scenario_c_successful_transfer(agent, env):
    result = agent.run("Transfer ₹10,000 to beneficiary BOB-001.")
    assert result.success is True
    assert result.reason_code is ReasonCode.OK
    assert result.action == "transfer"
    assert result.receipt_id == "TXN-SYN-0001"
    assert env.get_balance("ACC-SYN-001")["balance_inr"] == 240000
    assert env.get_history("ACC-SYN-001")["count"] == 1


def test_scenario_d_transfer_exceeding_limit(agent, env):
    result = agent.run("Transfer ₹60,000 to beneficiary BOB-001.")
    assert result.success is False
    assert result.reason_code is ReasonCode.LIMIT_EXCEEDED
    assert result.receipt_id is None
    assert env.get_balance("ACC-SYN-001")["balance_inr"] == 250000  # unchanged
    assert env.get_history("ACC-SYN-001")["transactions"][0]["status"] == "REJECTED"


def test_scenario_e_transfer_to_unverified_beneficiary(agent, env):
    result = agent.run("Transfer ₹5,000 to beneficiary BOB-002.")
    assert result.success is False
    assert result.reason_code is ReasonCode.BENEFICIARY_UNVERIFIED
    assert env.get_balance("ACC-SYN-001")["balance_inr"] == 250000


def test_scenario_f_insufficient_funds(agent, env):
    result = agent.run(
        "Transfer ₹10,000 to beneficiary BOB-001 from account ACC-SYN-002."
    )
    assert result.success is False
    assert result.reason_code is ReasonCode.INSUFFICIENT_FUNDS
    assert env.get_balance("ACC-SYN-002")["balance_inr"] == 5000


def test_unparseable_task(agent):
    result = agent.run("Order a pizza with extra cheese")
    assert result.success is False
    assert result.reason_code is ReasonCode.UNPARSEABLE_TASK
    assert result.tool_calls == []


def test_agent_runs_are_deterministic(env):
    r1 = RuleBasedAgent(BankEnvironment.from_dataset()).run(
        "Transfer ₹10,000 to beneficiary BOB-001."
    )
    r2 = RuleBasedAgent(BankEnvironment.from_dataset()).run(
        "Transfer ₹10,000 to beneficiary BOB-001."
    )
    assert r1.model_dump() == r2.model_dump()


def test_all_rejections_keep_ledger_auditable(agent, env):
    for task in (
        "Transfer ₹60,000 to beneficiary BOB-001.",
        "Transfer ₹5,000 to beneficiary BOB-002.",
    ):
        agent.run(task)
    hist = env.get_history("ACC-SYN-001")
    assert [t["status"] for t in hist["transactions"]] == ["REJECTED", "REJECTED"]


def test_state_matches_dataset_after_loading():
    # sanity: constructing env does not mutate loaded state
    state = load_dataset()
    env = BankEnvironment(state)
    assert env.fingerprint() == BankEnvironment(load_dataset()).fingerprint()
