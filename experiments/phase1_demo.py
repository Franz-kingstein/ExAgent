"""Phase 1 demo: run scenarios A-F through the banking simulator.

Usage:
    python experiments/phase1_demo.py            # deterministic rule-based agent
    python experiments/phase1_demo.py --llm      # LangChain + local Ollama agent

SYNTHETIC DATA ONLY (docs/safety.md).
"""

import os
import sys

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_xai.agent import make_agent
from trace_xai.banking.environment import BankEnvironment

SCENARIOS = [
    ("A. Balance check", "What is the balance of account ACC-SYN-001?"),
    ("B. Beneficiary verification", "Verify beneficiary BOB-001"),
    ("C. Successful transfer", "Transfer ₹10,000 to beneficiary BOB-001."),
    ("D. Over the limit", "Transfer ₹60,000 to beneficiary BOB-001."),
    ("E. Unverified beneficiary", "Transfer ₹5,000 to beneficiary BOB-002."),
    ("F. Insufficient funds", "Transfer ₹10,000 to beneficiary BOB-001 from account ACC-SYN-002."),
]


def main() -> None:
    use_llm = "--llm" in sys.argv
    env = BankEnvironment.from_dataset()
    agent = make_agent("llm" if use_llm else "rule", env)
    print(f"Agent: {agent.name}\n")

    for title, task in SCENARIOS:
        result = agent.run(task)
        status = "OK  " if result.success else "FAIL"
        print(f"[{status}] {title}")
        print(f"       task: {task}")
        print(f"       reason: {result.reason_code.value} | action: {result.action}")
        print(f"       message: {result.message}")
        if result.receipt_id:
            print(f"       receipt: {result.receipt_id}")
        print()

    print("Final state:")
    for acc in env.snapshot().account_ids():
        bal = env.get_balance(acc)["balance_inr"]
        hist = env.get_history(acc)
        print(f"  {acc}: balance INR {bal}, ledger entries {hist['count']}")


if __name__ == "__main__":
    main()
