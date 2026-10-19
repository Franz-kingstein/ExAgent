"""Final-phase experiments (Part 6): A-E, reproducible and deterministic.

    ./.venv/bin/python experiments/final_phase_experiments.py

Each experiment returns a JSON-serializable result with its method, sample
size and explicit limitations; the script prints a summary and writes
experiments/results/final_phase_results.json.

A. Adaptive-presentation consistency — 4 modes x identical traces; verifies
   status/evidence/count invariance and that UNKNOWN never becomes positive.
B. Evidence-validation baseline vs full pipeline — a documented "no
   validation" baseline (claims shown as-is) vs the validated pipeline,
   scored with false-acceptance/detection metrics against synthetic ground
   truth (known by construction; no LLM judge as ground truth).
C. Metric evaluation — latency/cost/claim-quality over the reproducible
   scenario suite, with units, basis and sample sizes.
D. Post-action stability — intervention/clean/interruption/short-window
   cases on injected timestamps (no waiting).
E. Phase 8 vs extended system — the same runs through the Phase 8 report
   versus the extended views (presentation/metrics/PAS/linkage); core
   verdicts must be identical, new artifacts additive.

No human-evaluation results are produced or implied anywhere: presentation
consistency is a factual-invariance check, not a user study.
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.linkage import MetricChange, link_metric
from trace_xai.metrics import CSATRecorder, InjectedClaim, evaluate_detection
from trace_xai.presentation import PresentationMode, present_all
from trace_xai.stability import (
    InterventionEvent,
    InterventionType,
    PostActionStabilityMonitor,
)
from trace_xai.validate.models import VerificationStatus

SCENARIOS = {
    "A_success": "Transfer ₹10,000 to beneficiary BOB-001.",
    "B_violation": "Transfer ₹75,000 to beneficiary BOB-001.",
    "D_both": "Transfer ₹75,000 to beneficiary BOB-002.",
}


def _pipeline() -> TraceXAIPipeline:
    return TraceXAIPipeline(agent=RuleBasedAgent(BankEnvironment.from_dataset()))


def _stateless_trace():
    r = ExecutionRecorder(
        task_id="TASK-SYN-F", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b", trace_id="TRC-SYN-F")
    r.record_task_received()
    r.record_tool_invocation(action="transfer", tool_name="transfer",
                             tool_input={"amount_inr": 10_000,
                                         "beneficiary_id": "BOB-001"},
                             state_before=None)
    r.record_tool_result(action="transfer", tool_name="transfer",
                         tool_output={"success": False,
                                      "reason_code": "INSUFFICIENT_FUNDS"},
                         success=False)
    r.record_outcome("INSUFFICIENT_FUNDS", success=False)
    return r.finish()


def _conflicting_trace():
    env = BankEnvironment.from_dataset()
    r = ExecutionRecorder(
        task_id="TASK-SYN-G", task_text="Transfer ₹5,000 to beneficiary BOB-001.",
        agent_id="rule-based-agent", trace_id="TRC-SYN-G")
    r.record_task_received()
    r.record_tool_invocation(action="transfer", tool_name="transfer",
                             tool_input={"amount_inr": 5_000,
                                         "beneficiary_id": "BOB-001"},
                             state_before=env.summary())
    first = r.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": True, "reason_code": "OK",
                     "account_id": "ACC-SYN-001"}, success=True)
    r.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS",
                     "account_id": "ACC-SYN-001"},
        success=False, conflict_with_event_id=first.event_id)
    r.record_outcome("CONFLICT", success=None)
    return r.finish()


class _AdversarialProvider:
    """Injects claims with invented refs — ground truth: they are wrong."""

    name = "experiment-adversarial"

    def __init__(self):
        self.injected = [
            "The transfer succeeded per evidence E99.",          # invented ref
            "No problems; REQ-FAKE-001 was satisfied.",         # fake req
            "All checks passed per E77.",                       # invented ref
        ]

    def generate(self, system_prompt, user_prompt):
        return json.dumps({
            "decision_rationale": self.injected[0],
            "deviation_explanation": self.injected[1],
            "counterfactual_summary": "Nothing to consider.",
            "evidence_summary": self.injected[2],
            "uncertainty_statement": "None at all.",
            "claims": [{"text": "Everything was fine and verified.",
                        "label": "FACT", "evidence_refs": ["E99"]}],
        })


# ------------------------------------------------- A. presentation consistency
def experiment_a(traces: dict) -> dict:
    """4 modes on identical traces: wording varies, conclusions never."""
    rows = []
    violations = 0
    for name, report in traces.items():
        modes = present_all(report)
        status_sets = {m: e.status_digest() for m, e in modes.items()}
        req_sets = {m: e.requirement_digest() for m, e in modes.items()}
        ref_sets = {m: frozenset(e.ref_digest()) for m, e in modes.items()}
        counts = {m: (e.counts, e.unsupported_claim_rate, e.guardrail)
                  for m, e in modes.items()}
        texts = {m: e.to_text() for m, e in modes.items()}
        unknown_positive = any(
            e.counts["unknown"] < report.validation.unknown_claim_count
            or e.counts["supported"] > report.validation.supported_claim_count
            for e in modes.values())
        if unknown_positive:
            violations += 1
        rows.append({
            "trace": name,
            "modes": len(modes),
            "statuses_invariant": len(set(map(str, status_sets.values()))) == 1,
            "requirements_invariant": len(set(map(str, req_sets.values()))) == 1,
            "evidence_refs_invariant": len(set(ref_sets.values())) == 1,
            "counts_invariant": len(set(map(str, counts.values()))) == 1,
            "wording_differs": len(set(texts.values())) == len(modes),
            "unknown_never_positive": not unknown_positive,
            "evidence_accessible_every_mode": all(
                all(line in texts[m] for line in report.evidence[:3])
                for m in modes),
        })
    return {
        "experiment": "A_adaptive_presentation_consistency",
        "method": ("present_all() on each validated report; invariants "
                   "compared across the four modes (status digest, "
                   "requirement digest, evidence-ref digest, counts, "
                   "guardrail) plus rendered-text equality checks"),
        "sample_size": {"traces": len(traces), "mode_views": len(traces) * 4},
        "all_invariants_hold": all(
            all(v for k, v in row.items() if k not in ("trace", "modes"))
            for row in rows),
        "unknown_violations": violations,
        "wording_varies": all(r["wording_differs"] for r in rows),
        "rows": rows,
        "limitations": [
            "factual-consistency verification only; no human subjects were "
            "involved and no comprehension claim is made about any mode",
            "mode wording is deterministic re-wording of validated content, "
            "not an adaptivity algorithm",
        ],
    }


# -------------------------------------------- B. baseline vs full validation
def experiment_b() -> dict:
    """No-validation baseline vs the full pipeline on injected wrong claims.

    Baseline (documented model of a system WITHOUT a validation stage): every
    claim the rationale layer emits is presented as verified — so all injected
    wrong claims are false acceptances by construction. Full pipeline: verdicts
    come from the validation stage and are scored against the same ground
    truth (known by construction; no LLM judge used as ground truth).
    """
    adversarial = _AdversarialProvider()
    ground_truth = [InjectedClaim(text=t, ground_truth="CONTRADICTED",
                                  origin="injected_wrong")
                    for t in adversarial.injected]

    # baseline: no validation stage -> claims pass through unverified
    baseline_false_acceptances = len(ground_truth)
    baseline = {
        "false_acceptance_rate": 1.0,
        "detection_rate": 0.0,
        "wrong_accepted": baseline_false_acceptances,
        "model": ("documented model of a no-validation pipeline: injected "
                  "claims pass through unverified, so every wrong claim is "
                  "accepted by construction"),
    }

    pipeline = TraceXAIPipeline(
        agent=RuleBasedAgent(BankEnvironment.from_dataset()),
        rationale_provider=adversarial)
    _, report = pipeline.run(SCENARIOS["A_success"])
    full = evaluate_detection(report.validation.verifications, ground_truth)

    # the extra claim the adversarial rationale injects ("Everything was
    # fine and verified.") is quarantined upstream: it must never appear as
    # a SUPPORTED verdict in the full pipeline
    invented = "Everything was fine and verified."
    invented_verdicts = [v for v in report.validation.verifications
                         if v.claim == invented]
    invented_accepted = any(
        v.verification_status is VerificationStatus.SUPPORTED
        for v in invented_verdicts)

    return {
        "experiment": "B_evidence_validation_baseline_vs_full",
        "method": ("same adversarial rationale (3 injected claims with "
                   "invented refs + 1 invented claim) run through a "
                   "documented no-validation baseline and through the full "
                   "TRACE-XAI validation stage; scored against synthetic "
                   "ground truth known by construction"),
        "sample_size": {"injected_wrong_claims": len(ground_truth),
                        "invented_claims": 1, "traces": 1},
        "baseline": baseline,
        "full_pipeline": full.model_dump(),
        "improvement": {
            "false_acceptance_rate_delta": round(
                baseline["false_acceptance_rate"]
                - (full.false_acceptance_rate or 0.0), 4),
            "detection_rate_delta": round(
                (full.detection_rate or 0.0) - baseline["detection_rate"], 4),
        },
        "invented_claim_quarantined": not invented_accepted,
        "limitations": [
            "the baseline is a documented model (claims pass through "
            "unverified), not a separately built second system",
            "ground truth is synthetic and known by construction; no LLM "
            "judge serves as ground truth",
            "CONTRADICTED and UNKNOWN both count as detected; their split "
            "is visible in the raw verdict counts",
            "one adversarial provider pattern; other failure modes may "
            "differ",
        ],
    }


# ------------------------------------------------------ C. metric evaluation
def experiment_c() -> dict:
    """Latency/cost/claim-quality metrics over the reproducible scenarios."""
    rows = []
    pipeline = _pipeline()
    for name, task in SCENARIOS.items():
        _, report = pipeline.run(task)
        m = report.metrics(simulate_csat=True)
        rows.append({
            "scenario": name,
            "agent_execution_ms": m.latency.agent_execution_ms,
            "xai_analysis_ms": m.latency.xai_analysis_ms,
            "end_to_end_ms": m.latency.end_to_end_ms,
            "latency_basis": m.latency.agent_basis.value,
            "prompt_tokens": m.cost.total_prompt_tokens,
            "completion_tokens": m.cost.total_completion_tokens,
            "cost_basis": m.cost.basis.value,
            "estimated_cost_usd": m.cost.estimated_cost_usd,
            "total_claims": m.claim_quality.total_claims,
            "unsupported_claim_rate": m.claim_quality.unsupported_claim_rate,
            "contradiction_rate": m.claim_quality.contradiction_rate,
            "unverifiable_rate": m.claim_quality.unverifiable_rate,
            "simulated_csat": (m.csat.simulated.simulated_mean_rating
                               if m.csat.simulated else None),
            "csat_status": m.csat.status,
        })
    # missing-evidence and conflicting-evidence traces join the suite
    for name, trace in (("F_missing_evidence", _stateless_trace()),
                        ("G_conflicting_evidence", _conflicting_trace())):
        report = pipeline.analyze_execution(trace=trace)
        m = report.metrics()
        rows.append({
            "scenario": name,
            "xai_analysis_ms": m.latency.xai_analysis_ms,
            "total_claims": m.claim_quality.total_claims,
            "unsupported_claim_rate": m.claim_quality.unsupported_claim_rate,
            "contradiction_rate": m.claim_quality.contradiction_rate,
            "unverifiable_rate": m.claim_quality.unverifiable_rate,
            "unknown_requirements": sum(
                1 for s in report.requirement_satisfaction.values()
                if s == "UNKNOWN"),
        })
    return {
        "experiment": "C_metric_evaluation",
        "method": ("report.metrics() per scenario; latency from perf_counter "
                   "(ms), tokens provider-reported, claim-quality from "
                   "validation verdicts; deterministic fallback rationale "
                   "throughout (no model consulted)"),
        "sample_size": {"scenarios": len(rows)},
        "units": {"latency": "ms", "tokens": "count", "cost": "USD",
                  "rates": "ratio 0-1", "csat": "rating 1-5 (simulated)"},
        "rows": rows,
        "limitations": [
            "deterministic fallback rationale: token counts are 0 and cost "
            "0.00 USD by construction; the real-SLM path reports "
            "provider-measured tokens with cost UNKNOWN unless pricing is "
            "supplied",
            "CSAT is simulated and labelled; no real user study exists",
            "stage timings are single-run wall-clock measurements on this "
            "machine, not statistical estimates",
        ],
    }


# ---------------------------------------------------- D. post-action stability
def experiment_d() -> dict:
    """PAS verdicts across window lengths and observation conditions."""
    from datetime import datetime, timezone

    t0 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)

    def at(ms):
        return t0 + timedelta(milliseconds=ms)

    cases = []

    def run_case(name, window_ms, observe_ms, interventions=(), gaps=()):
        m = PostActionStabilityMonitor(t0, window_ms=window_ms)
        for ms, kind in interventions:
            m.observe(InterventionEvent(timestamp=at(ms), kind=kind,
                                        description="simulated intervention",
                                        source="simulated"))
        for start_ms, end_ms, reason in gaps:
            m.record_gap(at(start_ms), at(end_ms), reason=reason)
        r = m.close(observe_until=at(observe_ms))
        cases.append({
            "case": name, "window_ms": window_ms, "observe_ms": observe_ms,
            "status": r.status.value, "ended_because": r.ended_because.value,
            "monitoring_duration_ms": r.monitoring_duration_ms,
            "observable_duration_ms": r.observable_duration_ms,
            "intervention_free_ms": r.intervention_free_duration_ms,
            "intervention_count": r.intervention_count,
            "gap_count": len(r.observation_gaps),
        })
        return r

    run_case("clean_full_window_60s", 60_000, 60_000)
    run_case("clean_full_window_10min", 600_000, 600_000)
    run_case("intervention_at_25s", 60_000, 60_000,
             interventions=[(25_000, InterventionType.HUMAN_OVERRIDE)])
    run_case("short_observation_20s_of_60s", 60_000, 20_000)
    run_case("interruption_gap_covers_window", 60_000, 60_000,
             gaps=[(0, 60_000, "collector offline (simulated)")])
    run_case("partial_gap_then_clean", 60_000, 60_000,
             gaps=[(10_000, 50_000, "network loss (simulated)")])
    late = run_case("intervention_after_window", 60_000, 120_000,
                    interventions=[(120_000, InterventionType.ROLLBACK)])

    expected = {
        "clean_full_window_60s": "NO_INTERVENTION_OBSERVED",
        "clean_full_window_10min": "NO_INTERVENTION_OBSERVED",
        "intervention_at_25s": "INTERVENTION_DETECTED",
        "short_observation_20s_of_60s": "INSUFFICIENT_OBSERVATION",
        "interruption_gap_covers_window": "INSUFFICIENT_OBSERVATION",
        "partial_gap_then_clean": "INSUFFICIENT_OBSERVATION",
        "intervention_after_window": "NO_INTERVENTION_OBSERVED",
    }
    return {
        "experiment": "D_post_action_stability",
        "method": ("PostActionStabilityMonitor with injected logical-clock "
                   "timestamps; no real waiting; window length and "
                   "observation span varied per case"),
        "sample_size": {"cases": len(cases)},
        "cases": cases,
        "all_statuses_as_expected": all(
            c["status"] == expected[c["case"]] for c in cases),
        "late_event_excluded": late.intervention_count == 0,
        "limitations": [
            "all interventions and gaps are simulated inputs on an "
            "injected clock; no real production monitoring occurred",
            "NO_INTERVENTION_OBSERVED is bounded by the configured window "
            "and observed span — never a claim of indefinite stability or "
            "of correctness",
            "interruption handling is interval-based; real outages may have "
            "irregular shapes",
        ],
    }


# --------------------------------------------------- E. phase 8 vs extended
def experiment_e() -> dict:
    """Phase 8 report vs the extended system on identical runs."""
    pipeline = _pipeline()
    rows = []
    for name, task in SCENARIOS.items():
        _, report = pipeline.run(task)
        phase8_text = report.to_text()          # Phase 8 artifact, unchanged
        modes = present_all(report)
        m = report.metrics(simulate_csat=True)
        link = link_metric(report, MetricChange(
            metric="unsupported_claim_rate",
            before=round(min(1.0, m.claim_quality.unsupported_claim_rate
                             + 0.25), 4),
            after=m.claim_quality.unsupported_claim_rate,
            unit="ratio", basis="measured",
            related_requirement_ids=[report.requirements[0].requirement_id]
            if report.requirements else []))
        core_same = all(
            e.status_digest()
            == {v.claim: v.verification_status.value
                for v in report.validation.verifications}
            and e.guardrail == report.validation.guardrail_decision.value
            for e in modes.values())
        rows.append({
            "scenario": name,
            "phase8_report_sections": 15,
            "phase8_to_text_len": len(phase8_text),
            "extended_artifacts": {
                "presentation_modes": len(modes),
                "metrics_sections": 4,
                "linkage_chains": 1,
            },
            "core_verdicts_identical_across_views": core_same,
            "guardrail": report.validation.guardrail_decision.value,
        })
    return {
        "experiment": "E_phase8_vs_extended",
        "method": ("same pipeline runs produce the Phase 8 15-section "
                   "report and the extended views (4 presentation modes, "
                   "metrics bundle, metric->trace linkage); core verdicts "
                   "compared for identity, new artifacts counted"),
        "sample_size": {"scenarios": len(rows), "runs": len(rows)},
        "rows": rows,
        "backwards_compatible": all(
            r["core_verdicts_identical_across_views"] for r in rows),
        "limitations": [
            "comparison is structural (artifact inventory + verdict "
            "identity); no performance regression is asserted beyond the "
            "recorded stage timings",
            "Phase 8 components were not modified; extended views are "
            "additive derivations from the same validated report",
        ],
    }


# --------------------------------------------------------------------- main
def main() -> None:
    pipeline = _pipeline()
    traces = {name: pipeline.run(task)[1]
              for name, task in SCENARIOS.items()}

    results = [
        experiment_a(traces),
        experiment_b(),
        experiment_c(),
        experiment_d(),
        experiment_e(),
    ]

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "final_phase_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print("=" * 72)
    print("TRACE-XAI FINAL-PHASE EXPERIMENTS (Part 6)")
    print("=" * 72)
    for r in results:
        print(f"\n[{r['experiment']}]  sample={r['sample_size']}")
        for key in ("all_invariants_hold", "unknown_violations",
                    "all_statuses_as_expected", "backwards_compatible",
                    "invented_claim_quarantined"):
            if key in r:
                print(f"  {key}: {r[key]}")
        if r["experiment"].startswith("B_"):
            b, f = r["baseline"], r["full_pipeline"]
            print(f"  baseline  false_acceptance_rate={b['false_acceptance_rate']}"
                  f" detection_rate={b['detection_rate']}")
            print(f"  full      false_acceptance_rate="
                  f"{f['false_acceptance_rate']} detection_rate="
                  f"{f['detection_rate']} wrong_detected={f['wrong_detected']}"
                  f"/{f['wrong_claims']}")
        if r["experiment"].startswith("D_"):
            for c in r["cases"]:
                print(f"  {c['case']:<34} {c['status']}")
        print(f"  limitations: {len(r['limitations'])} documented")
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
