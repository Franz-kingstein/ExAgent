"""Phase 4 demo: hybrid behavior-deviation explainability (Q2).

Requirement-violation detection (Phase 3) + interpretable trajectory anomaly
detection, combined into one deviation explanation.

Usage:
    python experiments/phase4_demo.py
"""

import os
import sys

os.environ.setdefault("USE_TORCH", "0")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.explain import DeviationExplainer

STATE_OK = {
    "balances": {"ACC-SYN-001": 250000},
    "per_txn_limits": {"ACC-SYN-001": 50000},
    "beneficiaries_verified": {"BOB-001": True},
}


def over_limit_retry_trace():
    """Controlled anomaly: 3 attempts of an over-limit transfer."""
    rec = ExecutionRecorder(
        task_id="TASK-DEMO-0004", task_text="Transfer ₹60,000 to beneficiary BOB-001.",
        agent_id="manual-agent", trace_id="TRC-DEMO-0004",
    )
    rec.record_task_received()
    for _ in range(3):
        inv = rec.record_tool_invocation("TRANSFER", "transfer",
                                         {"beneficiary_id": "BOB-001", "amount_inr": 60000},
                                         state_before=STATE_OK)
        rec.record_tool_result("TRANSFER", "transfer",
                               {"success": False, "reason_code": "LIMIT_EXCEEDED"},
                               state_after=STATE_OK, success=False,
                               related_event_id=inv.event_id)
    rec.record_outcome("LIMIT_EXCEEDED", success=False)
    return rec.finish()


def main() -> None:
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    explainer = DeviationExplainer()

    scenarios = [
        ("Normal trajectory", agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")[1]),
        ("Requirement violation only",
         agent.run_traced("Transfer ₹60,000 to beneficiary BOB-001.")[1]),
        ("Anomaly + violation (controlled)", over_limit_retry_trace()),
    ]

    for title, trace in scenarios:
        deviation = explainer.explain(trace)
        print("=" * 78)
        print(f"{title}  [{trace.trace_id}]")
        print(f" TASK: {trace.task_text}")
        print(f" CATEGORY: {deviation.category.value} | RELATION: {deviation.relation.value}")
        for v in deviation.requirement_violations:
            print(f"  VIOLATION: {v.requirement_id} — {v.explanation.split(' Evidence:')[0]}")
        for a in deviation.anomalies:
            print(f"  ANOMALY: {a.anomaly_id} {a.anomaly_type.value} "
                  f"[{a.severity.value}, conf {a.confidence:.2f}] {a.explanation}")
        print(f" EXPLANATION:\n{deviation.combined_explanation}")


if __name__ == "__main__":
    main()
