"""Phase 3 demo: decision explainability via requirement-to-action matching.

Runs banking scenarios, then explains each decision against the requirement
registry using observable trace evidence only (no LLM).

Usage:
    python experiments/phase3_demo.py
"""

import os
import sys

os.environ.setdefault("USE_TORCH", "0")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.explain import DecisionExplainer

SCENARIOS = [
    "Transfer ₹10,000 to beneficiary BOB-001.",
    "Transfer ₹5,000 to beneficiary BOB-002.",
    "Transfer ₹60,000 to beneficiary BOB-002.",
]


def main() -> None:
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    explainer = DecisionExplainer()

    for task in SCENARIOS:
        _result, trace = agent.run_traced(task)
        explanation = explainer.explain(trace)
        print("=" * 78)
        print(f"TASK: {task}")
        print(f"TRACE: {trace.trace_id} | ACTION: {explanation.action}")
        for a in explanation.requirements:
            print(f"  [{a.status.value:9}] {a.requirement_id} "
                  f"(confidence {a.confidence:.2f}) — {a.requirement_text}")
            print(f"             {a.explanation}")
        print(f"RATIONALE: {explanation.rationale}")


if __name__ == "__main__":
    main()
