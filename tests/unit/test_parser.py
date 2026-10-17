"""Parser tests: natural-language task -> typed Intent (deterministic)."""

from trace_xai.agent.parser import IntentType, parse_task


def test_parse_transfer_primary_scenario():
    intent = parse_task("Transfer ₹10,000 to beneficiary BOB-001.")
    assert intent.type is IntentType.TRANSFER
    assert intent.amount_inr == 10000
    assert intent.beneficiary_id == "BOB-001"
    assert intent.account_id is None  # defaults to environment's account


def test_parse_transfer_with_account_and_plain_amount():
    intent = parse_task("transfer 2500 to BOB-002 from account ACC-SYN-002")
    assert intent.type is IntentType.TRANSFER
    assert intent.amount_inr == 2500
    assert intent.beneficiary_id == "BOB-002"
    assert intent.account_id == "ACC-SYN-002"


def test_parse_transfer_rejects_decimal_amount():
    intent = parse_task("Transfer ₹10,000.50 to beneficiary BOB-001")
    assert intent.type is IntentType.TRANSFER
    assert intent.amount_inr is None
    assert intent.note is not None


def test_parse_balance():
    intent = parse_task("What is the balance of account ACC-SYN-001?")
    assert intent.type is IntentType.BALANCE
    assert intent.account_id == "ACC-SYN-001"


def test_parse_beneficiary_check():
    intent = parse_task("Verify beneficiary BOB-001")
    assert intent.type is IntentType.BENEFICIARY_CHECK
    assert intent.beneficiary_id == "BOB-001"


def test_parse_history():
    intent = parse_task("Show transaction history for ACC-SYN-001")
    assert intent.type is IntentType.HISTORY
    assert intent.account_id == "ACC-SYN-001"


def test_parse_unknown():
    intent = parse_task("Order a pizza with extra cheese")
    assert intent.type is IntentType.UNKNOWN
    assert intent.note is not None
