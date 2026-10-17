"""Phase 6 demo: SLM rationale layer (user-facing, evidence-grounded).

Pipeline: agent trace -> Q1 decision + Q2 deviation + Q3 counterfactual
(deterministic) -> structured evidence packet -> SLM (local Ollama, thinking
disabled) -> five-section rationale with citation grounding. Falls back to
deterministic composition if no model is reachable.

Usage: ./.venv/bin/python experiments/phase6_demo.py
"""

import os
import sys
import time
import resource

os.environ.setdefault("USE_TORCH", "0")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.rationale import (
    OllamaRationaleProvider,
    RationaleGenerator,
    build_context,
    render_prompt,
)


def ollama_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def show(title: str, rationale) -> None:
    print(f"\n=== {title} ===")
    print(f"provider: {rationale.provider}  llm_used: {rationale.llm_used}  "
          f"grounded: {rationale.grounded}")
    print(f"[decision_rationale]     {rationale.decision_rationale}")
    print(f"[deviation_explanation]  {rationale.deviation_explanation}")
    print(f"[counterfactual_summary] {rationale.counterfactual_summary}")
    print(f"[evidence_summary]       {rationale.evidence_summary}")
    print(f"[uncertainty_statement]  {rationale.uncertainty_statement}")
    for claim in rationale.claims:
        print(f"  claim [{claim.label.value}] {claim.text}  refs={claim.evidence_refs}")
    if rationale.unsupported_claims:
        for bad in rationale.unsupported_claims:
            print(f"  QUARANTINED: {bad}")
    for note in rationale.notes:
        print(f"  note: {note}")


def main() -> None:
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    provider = OllamaRationaleProvider() if ollama_up() else None
    generator = RationaleGenerator(provider=provider)
    print(f"provider: {provider.name if provider else 'none (deterministic fallback)'}")

    # --- evidence-complete: over-limit transfer (Q1 violation + Q3 counterfactual)
    _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
    ctx = build_context(trace)
    system, user = render_prompt(ctx)
    print(f"\nprompt sizes: system={len(system)} chars, packet={len(user)} chars, "
          f"evidence items={len(ctx.evidence)}")

    t0 = time.perf_counter()
    rationale = generator.generate(trace, context=ctx)
    elapsed = time.perf_counter() - t0
    show("complete evidence (real SLM)", rationale)
    if provider is not None and provider.last_usage:
        usage = provider.last_usage
        print(f"\nlatency: {elapsed:.2f}s total, model eval {usage.get('eval_count')} tokens, "
              f"prompt {usage.get('prompt_eval_count')} tokens, "
              f"duration {usage.get('total_duration_ns', 0) / 1e9:.2f}s")
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak RSS of this demo process: {rss_mb:.0f} MiB (model weights live in the "
          f"Ollama daemon, not this process)")

    # --- evidence missing: LLM-style trace without recorded state
    show("missing evidence (SLM must admit UNKNOWN)",
         generator.generate(_stateless_trace()))


def _stateless_trace():
    from trace_xai.core.recorder import ExecutionRecorder

    recorder = ExecutionRecorder(
        task_id="task-llm-style", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b",
    )
    recorder.record_task_received()
    recorder.record_tool_invocation(
        action="transfer", tool_name="transfer",
        tool_input={"amount_inr": 10_000, "beneficiary_id": "BOB-001"},
        state_before=None,
    )
    recorder.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS"},
        success=False,
    )
    recorder.record_outcome("INSUFFICIENT_FUNDS", success=False)
    return recorder.finish()


if __name__ == "__main__":
    main()
