"""Generate the committed sample trace (Phase 2).

Produces data/traces/sample_transfer_success.json deterministically: the same
transfer task always yields byte-identical JSON, so the artifact is reviewable
in git diffs and usable by TRACE-XAI without re-running the agent.

Usage:
    python experiments/phase2_sample_trace.py
"""

import os
import sys

os.environ.setdefault("USE_TORCH", "0")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_xai.agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment

OUTPUT_PATH = ROOT / "data" / "traces" / "sample_transfer_success.json"


def main() -> None:
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    _result, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(trace.to_json() + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(trace.events)} events, outcome {trace.final_outcome})")


if __name__ == "__main__":
    main()
