"""Phase 5 demo: counterfactual explainability (Q3).

Runs the deterministic rule agent on the synthetic banking scenarios, then
asks the CounterfactualExplainer what could have happened under minimal
alternative actions. Simulation predictions and controlled replays are
reported separately — never blended.

Usage: ./.venv/bin/python experiments/phase5_demo.py
"""

import json
import os
import sys

os.environ.setdefault("USE_TORCH", "0")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.explain.counterfactual import CounterfactualExplainer
from trace_xai.explain.models import ActionSpec


def show(title: str, explanation) -> None:
    print(f"\n=== {title} ===")
    print(explanation.actual_summary)
    for note in explanation.notes:
        print(f"  [note] {note}")
    if not explanation.counterfactuals:
        print("  (no counterfactuals)")
        return
    for cf in explanation.counterfactuals:
        print(f"\n  {cf.cf_id} [{cf.source}] {cf.changed_condition}")
        o, a = cf.original_action, cf.alternative_action
        print(f"    actual:      transfer INR {o.amount_inr} to {o.beneficiary_id}")
        print(f"    alternative: transfer INR {a.amount_inr} to {a.beneficiary_id}")
        print(f"    predicted:   {'success' if cf.predicted_outcome.success else cf.predicted_outcome.reason_code}")
        if cf.predicted_outcome.expected_state_change:
            print(f"    state:       {cf.predicted_outcome.expected_state_change}")
        if cf.replay and cf.replay.performed:
            status = "success" if cf.replay.execution_success else cf.replay.reason_code
            print(f"    replayed:    yes ({status}, txn {cf.replay.transaction_id})")
        else:
            print("    replayed:    no (prediction only)")
        print(f"    impact:      {cf.impact.value}   uncertainty: {cf.uncertainty}")
        print(f"    explain:     {cf.explanation}")


def main() -> None:
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    explainer = CounterfactualExplainer()

    # 1+3: over-limit transfer -> minimal clamp satisfies, improves outcome
    _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
    show("Q3: over-limit transfer (satisfies + improves)", explainer.explain(trace))

    # 2: low-balance account -> clamp-to-limit still fails on funds
    _, trace = agent.run_traced(
        "Transfer ₹20,000 to beneficiary BOB-001 from account ACC-SYN-002."
    )
    show("Q3: double violation (alternative still fails)", explainer.explain(trace))

    # 4: provided larger amount worsens a clean success
    _, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
    show(
        "Q3: provided alternative worsens outcome",
        explainer.explain(
            trace,
            alternatives=[ActionSpec(amount_inr=60_000, beneficiary_id="BOB-001")],
        ),
    )

    # 5: replay disabled -> simulation only, higher uncertainty
    _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
    no_replay = CounterfactualExplainer(replay_enabled=False)
    show("Q3: replay unavailable (simulation only)", no_replay.explain(trace))

    # machine-readable export of the first explanation
    _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
    payload = explainer.explain(trace).model_dump(mode="json")
    print("\n=== JSON export (truncated) ===")
    print(json.dumps(payload, indent=2)[:1200] + "\n  ...")


if __name__ == "__main__":
    main()
