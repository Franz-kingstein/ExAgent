"""Phase 8 demo: the integrated TRACE-XAI pipeline.

Runs scenarios A-F through analyze_execution/run and prints the final
15-section XAI report for one representative case, plus per-stage timings.

    ./.venv/bin/python experiments/phase8_demo.py

Deterministic and fast (<1 s): the rationale layer uses its built-in
deterministic fallback. Add rationale_provider=OllamaRationaleProvider() to
exercise the real SLM path (~20 s per report; needs the Ollama service).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder


def scenario_c_trace():
    """Repeated invalid-amount failures: trajectory anomaly, no violation."""
    env = BankEnvironment.from_dataset()
    r = ExecutionRecorder(task_id="TASK-SYN-C",
                          task_text="Transfer an invalid amount.",
                          agent_id="rule-based-agent", trace_id="TRC-SYN-C")
    r.record_task_received()
    for _ in range(2):
        r.record_tool_invocation(action="transfer", tool_name="transfer",
                                 tool_input={"amount_inr": -5000,
                                             "beneficiary_id": "BOB-001"},
                                 state_before=env.summary())
        r.record_tool_result(action="transfer", tool_name="transfer",
                             tool_output={"success": False,
                                          "reason_code": "INVALID_AMOUNT",
                                          "account_id": "ACC-SYN-001"},
                             success=False)
    r.record_outcome("INVALID_AMOUNT", success=False)
    return r.finish()


def main():
    # fresh pipeline per scenario: banking environments mutate on success
    new_pipeline = lambda: TraceXAIPipeline(  # noqa: E731
        agent=RuleBasedAgent(BankEnvironment.from_dataset()))

    print("=" * 72)
    print("PHASE 8: integrated pipeline (deterministic rationale fallback)")
    print("=" * 72)

    scenarios = [
        ("A. successful action", lambda p: p.run(
            "Transfer ₹10,000 to beneficiary BOB-001.")),
        ("B. requirement violation", lambda p: p.run(
            "Transfer ₹75,000 to beneficiary BOB-001.")),
        ("C. trajectory anomaly", lambda p: p.analyze_execution(
            trace=scenario_c_trace())),
        ("E. counterfactual improvement (75k)", lambda p: p.run(
            "Transfer ₹75,000 to beneficiary BOB-001.")),
    ]

    for name, run in scenarios:
        t0 = time.perf_counter()
        if "anomaly" in name:
            result, report = None, run(new_pipeline())
        else:
            result, report = run(new_pipeline())
        wall = (time.perf_counter() - t0) * 1000
        stages = ", ".join(f"{k}={v:.1f}ms"
                           for k, v in report.timings_ms.items())
        outcome = "recorded trace" if result is None else (
            "success" if result.success else f"rejected ({result.reason_code})")
        print(f"\n[{name}]")
        print(f"  action={report.action} {outcome}")
        print(f"  satisfaction: {report.requirement_satisfaction}")
        print(f"  deviations: {report.deviation.category.value} | "
              f"anomalies: {len(report.deviation.anomalies)} | "
              f"counterfactuals: "
              f"{len(report.counterfactual_explanation.counterfactuals)}")
        print(f"  validation: {report.validation.overall_validation_status.value}"
              f" (rate {report.validation.unsupported_claim_rate:.2f}) "
              f"guardrail={report.validation.guardrail_decision.value} "
              f"[{report.validation.supported_claim_count}S/"
              f"{report.validation.contradicted_claim_count}C/"
              f"{report.validation.unknown_claim_count}U]")
        print(f"  wall={wall:.1f}ms | {stages}")

    # full 15-section report for scenario B
    _, report = new_pipeline().run("Transfer ₹75,000 to beneficiary BOB-001.")
    print("\n\n" + report.to_text())
    print("\n[extra] embedded counterfactuals (audit detail):")
    for cf in report.counterfactual_explanation.counterfactuals:
        print(f"  {cf.summary()}")


if __name__ == "__main__":
    main()
