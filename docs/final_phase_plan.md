# Final Phase — Implementation Plan (inspected from actual repo state)

> **STATUS: COMPLETE.** All 7 increments delivered and committed
> (`bef55c7` → `616d0a6` + docs commit); 312 tests passing (unit + e2e +
> integration incl. real-model tests); experiments A–E green with results
> in `experiments/results/final_phase_results.json`; GUI launch verified.
> The sections below are kept as the original inspected baseline.

## Verified baseline (2026-10-19)

- HEAD `d2c7068` "Phase 8", working tree **clean**, branch `main` == `origin/main`.
- **188 tests pass**: `tests/unit` + `tests/e2e` = **176 passed in 1.47s**;
  `tests/integration` = **12 passed in 116.67s** (exit 0).
- GUI: **none exists**. Only `docs/phases.md:60` — "Demo UI (FastAPI/Gradio)
  decision deferred until after Phase 8."
- Available already installed (no new deps needed): gradio 5.38.0, flask 2.2.5,
  fastapi 0.142.2, uvicorn, pandas 2.3.3, matplotlib 3.10.8, jinja2 3.1.6,
  pydantic 2.12.5, pytest 9.0.2, `time-machine` (simulated clock available).
- Metrics support today: **none**. `FinalXAIReport.to_text()` is the only export
  (no JSON export). No CSAT, no cost, no stability monitoring.
- Timing today: `FinalXAIReport.timings_ms` = {decision, deviation,
  counterfactual, rationale, validation} — XAI analysis only. **Agent execution
  latency is NOT timed** (`pipeline.run()` calls `run_traced` unmeasured).
- Provider usage already captured: `OllamaRationaleProvider.last_usage` =
  {eval_count, prompt_eval_count, total_duration_ns} → real token counts.
- Validation already yields supported/contradicted/unknown counts,
  `unsupported_claim_rate = (contradicted + unknown) / total`.

## Gaps vs. requested features

| Feature | Status |
|---|---|
| 4 presentation modes | missing |
| Latency: agent vs XAI vs end-to-end | partially (agent missing) |
| Cost (tokens/pricing) | missing (usage data available) |
| Unsupported-claim rate w/ documented formula | present, undocumented |
| CSAT feedback | missing |
| Post-Action Stability (PAS) | missing |
| Metric → trace evidence linkage | missing |
| Experiments / baseline comparison | partial (phase demos only) |
| GUI | missing |
| Report export (JSON/MD) | missing |

## Increments (one at a time, tested before moving on)

1. **Presentation modes** — `presentation/` package; all 4 modes from the same
   `FinalXAIReport`; statuses + evidence refs invariant.
2. **Metrics** — `metrics/` package: latency (add agent timing to pipeline),
   cost (real usage, pricing = unknown when absent), hallucination (documented
   formula), CSAT (documented formula, not a real user study).
3. **Post-Action Stability** — `metrics/stability.py`, simulated clock,
   configurable window, 3 statuses, interruption handling.
4. **Metric→trace linkage** — `metrics/linkage.py`, the 6-step chain, causal vs
   association labelling, claims verified through the existing validator.
5. **Experiments** — `experiments/final_evaluation.py` (A–E), deterministic.
6. **GUI** — Gradio app extending the architecture (not rebuilding).
7. **Docs** — metric definitions, innovation summary, run instructions.

## Non-negotiables carried over

Synthetic data only; no real banking; no fabricated costs/ratings/citations;
UNKNOWN never becomes SUPPORTED via simplification; do not weaken existing
tests; no remote push without explicit approval.
