"""Phase 7 demo: explanation validation (Q4).

Pipeline: claim extraction -> evidence retrieval -> deterministic rules ->
lexical NLI -> optional LLM judge -> guardrail decision. Runs the six
validation scenarios against the synthetic banking trace and measures
deterministic throughput (+ one real judge call if Ollama is up).

Usage: ./.venv/bin/python experiments/phase7_demo.py
"""

import os
import sys
import time

os.environ.setdefault("USE_TORCH", "0")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.rationale import OllamaRationaleProvider, build_context
from trace_xai.rationale.models import RationaleClaim, StructuredRationale
from trace_xai.validate import ExplanationValidator


def ollama_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def make_rationale(claims):
    return StructuredRationale(
        task_id="TASK-SYN-0001", trace_id="TRC-SYN-0001",
        agent_id="rule-based-agent", action="TRANSFER",
        decision_rationale="Section.", deviation_explanation="Section.",
        counterfactual_summary="Section.", evidence_summary="Section.",
        uncertainty_statement="Section.",
        claims=[RationaleClaim(**c) for c in claims],
        provider="demo",
    )


def show(title, report):
    print(f"\n=== {title} ===")
    for v in report.verifications:
        print(f"  [{v.verification_status.value:12s}] ({v.verification_method.value}, "
              f"conf {v.confidence:.2f}) {v.claim}")
        print(f"      -> {v.explanation}")
    print(f"  supported={report.supported_claim_count} "
          f"contradicted={report.contradicted_claim_count} "
          f"unknown={report.unknown_claim_count} "
          f"rate={report.unsupported_claim_rate:.2f} "
          f"overall={report.overall_validation_status.value} "
          f"guardrail={report.guardrail_decision.value}")


def ref_for(ctx, substring):
    return next(e.ref for e in ctx.evidence if substring in e.field)


def main():
    agent = RuleBasedAgent(BankEnvironment.from_dataset())
    _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
    ctx = build_context(trace)
    validator = ExplanationValidator()  # deterministic-only by default

    amt, lim = ref_for(ctx, "amount_inr"), ref_for(ctx, "per_txn_limits")
    ben, bal = ref_for(ctx, "beneficiaries_verified"), ref_for(ctx, "balances")

    scenarios = [
        ("1. correct", [
            {"text": "The transfer amount of INR 75,000 exceeded the per-transaction "
                     "limit.", "label": "EVIDENCE", "evidence_refs": [amt, lim]},
            {"text": "REQ-AMT-001 was violated.", "label": "EVIDENCE",
             "evidence_refs": ["REQ-AMT-001"]},
        ]),
        ("4. contradicted (spec example)", [
            {"text": "The transfer was within the ₹50,000 limit.", "label": "FACT",
             "evidence_refs": [amt, lim]},
        ]),
        ("5. missing evidence (spec example)", [
            {"text": "The agent transferred ₹75,000 because the beneficiary was "
                     "verified.", "label": "EVIDENCE", "evidence_refs": [amt]},
        ]),
        ("3. unsupported (invented + uncited)", [
            {"text": "The transfer was processed instantly.", "label": "FACT",
             "evidence_refs": []},
            {"text": "The environment confirmed everything.", "label": "FACT",
             "evidence_refs": ["E99"]},
        ]),
        ("2. partially correct", [
            {"text": "The beneficiary was verified.", "label": "EVIDENCE",
             "evidence_refs": [ben]},
            {"text": "The agent deliberated carefully before choosing.", "label":
             "INFERENCE", "evidence_refs": [bal]},
        ]),
    ]

    reports = []
    for title, claims in scenarios:
        report = validator.validate(make_rationale(claims), ctx)
        reports.append(report)
        show(title, report)

    # throughput of the deterministic pipeline
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        validator.validate(make_rationale(scenarios[0][1]), ctx)
    per = (time.perf_counter() - t0) / n * 1000
    print(f"\ndeterministic validation: {per:.3f} ms per report ({n} runs)")

    if ollama_up():
        print("\n[optional] one real LLM-judge call on a semantic paraphrase...")
        judge_validator = ExplanationValidator(
            judge_provider=OllamaRationaleProvider(model="qwen3.5:4b"))
        t0 = time.perf_counter()
        report = judge_validator.validate(make_rationale([
            {"text": "The payee had been approved for receiving transfers.",
             "label": "INFERENCE", "evidence_refs": [ben]},
        ]), ctx)
        show("judge scenario", report)
        print(f"judge latency: {time.perf_counter() - t0:.1f}s "
              f"(judge_used={report.judge_used})")
    else:
        print("\nOllama not reachable: deterministic-only run (judge skipped)")


if __name__ == "__main__":
    main()
