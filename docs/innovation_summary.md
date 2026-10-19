# TRACE-XAI — Innovation & Contribution Summary (working title, not a novelty claim)

**Title status:** "TRACE-XAI" is a working project title. It is not a claim
of a new research field, and nothing in this document should be read as
claiming priority for an established research area.

This document separates what TRACE-XAI **reuses**, what it **integrates
newly**, what it **adds as new features**, what its experiments actually
**demonstrate**, and what remains **unverified**. A literature review was
run online while preparing this document (sources listed at the end); had
the network been unavailable, all novelty statements would be marked
unverified instead.

---

## 1. Reused methods (established in prior work — not claimed as novel)

| Method | Where TRACE-XAI uses it | Prior-art status |
|---|---|---|
| Audience-tailored / adaptive explanation presentation | `presentation/` (4 modes) | Well-established research direction: adaptive/XAI interfaces (e.g. Adaptive XAI workshops at IUI 2024/2025, AXAI), "Explanation is Not a Technical Term" (arXiv 2207.00007) argues explicitly for audience-dependent explanation forms, contextualized XAI evaluation (HCOMP 2022). TRACE-XAI does **not** claim audience-adaptive wording as a novel algorithm — it is a deterministic re-wording layer. |
| Rule-based requirement/constraint checking as explanation backbone | `requirements/`, `explain/matcher.py` | Standard in symbolic XAI and runtime verification. |
| Counterfactual explanations + replay | `explain/counterfactual.py` | Counterfactual explanations are a mature XAI genre (Wachter et al. tradition). |
| LLM/SLM-as-judge for claim verification | `validate/judge.py` (capped, advisory) | Common practice; TRACE-XAI deliberately limits it (never overrules deterministic rules, confidence capped) rather than inventing a new judging method. |
| Hallucination/unsupported-claim rate as an evaluation metric | `metrics/` | Widely used in LLM evaluation; TRACE-XAI's contribution here is *decomposition and labelling* (contradicted vs unknown), not the metric's existence. |
| Post-deployment monitoring concept | `stability/` | NIST AI 800-4 (2026) documents post-deployment monitoring as an established need with nascent best practice; agent-stability/drift metrics exist (e.g. Agent Stability Index work). |
| CSAT/retention as satisfaction metrics | `metrics/` | Standard UX metrics; TRACE-XAI runs **no** real study and labels its simulated values accordingly. |

## 2. New integration (composed here; components individually are not new)

1. **One validated report, four invariant audience views.** The wording
   layer is a *pure deterministic function* of the validated
   `FinalXAIReport`: statuses, counts, guardrail and evidence references
   are copied, never recomputed. This is what mechanically guarantees
   (and is tested to guarantee) that simplifying wording can never turn
   UNKNOWN into SUPPORTED. The novelty claim, if any, is in the
   *integration property and its invariance tests*, not in having modes.
2. **Metric → evidence chain linkage.** METRIC → OBSERVED CHANGE →
   RELEVANT TRACE EVENTS → EXPLANATION CLAIM → EVIDENCE REFERENCES →
   VALIDATION STATUS, labelled ASSOCIATION/HYPOTHESIS and never causal.
   Traceability-of-metrics is an active area (cf. EU AI Act traceability
   discussions, causal-attribution-for-agentic-decisions preprints);
   TRACE-XAI's chain is a conservative, association-only construction
   over its own validated artifacts.
3. **Explainability metrics attached to the same artifacts as the
   explanation.** Latency split (agent vs XAI vs end-to-end), real token
   cost, decomposed claim quality, labelled CSAT — all derived from one
   report object, so numbers and claims cannot drift apart.

## 3. New features (implemented in this project, checked in)

- `presentation/` — 4-mode audience presentation with FACT/INFERENCE/
  UNKNOWN tagging and cross-mode invariance digests (24 tests).
- `metrics/` — latency/cost/claim-quality/CSAT bundles with unit + basis
  on every number; `CSATRecorder` (records only what a person enters);
  `evaluate_detection` for ground-truth scoring (23 tests).
- `stability/` — bounded post-action stability monitoring with explicit
  window, intervention types, gaps, end-reason and the three statuses;
  injected clock, zero waiting (24 tests).
- `linkage/` — association-only metric→trace chains (12 tests).
- `app.py` — Gradio evaluator GUI, 7 tabs / 12 capabilities, every panel
  wired to real handlers (28 tests).
- `experiments/final_phase_experiments.py` — reproducible experiments
  A–E with documented limitations (13 regression tests).

## 4. Demonstrated findings (what the experiments actually show)

All findings are on **synthetic data only**, with deterministic
fallbacks unless stated otherwise (see
`experiments/results/final_phase_results.json`):

1. **Presentation consistency (Exp. A):** across 3 traces × 4 modes
   (12 views), statuses, requirement digests, evidence refs, counts and
   guardrail were identical in every mode; wording differed; UNKNOWN
   counts never shrank in any mode. *Factual invariance only — no
   comprehension claim; no human subjects.*
2. **Validation value vs baseline (Exp. B):** with 3 injected wrong
   claims + 1 invented claim, the documented no-validation baseline
   false-accepts 100% (by construction of the baseline model); the full
   pipeline detected 3/3 (false-acceptance rate 0.0, detection 1.0),
   and the invented claim was quarantined before validation.
   *Baseline is a documented model, not a second implemented system.*
3. **Metrics behave honestly (Exp. C):** 5 scenarios; deterministic
   fallback → 0 tokens / 0.00 USD by construction; missing-evidence
   scenario surfaces UNKNOWN requirements without contradiction.
4. **PAS verdicts are window-bounded (Exp. D):** 7/7 cases matched
   expectations, including interruption → INSUFFICIENT_OBSERVATION and
   an event after the window excluded from the verdict.
5. **Backward compatibility (Exp. E):** Phase 8 report artifacts are
   unchanged; extended views derive identical core verdicts from the
   same runs.

## 5. Unverified / not claimed

- **Human-factors claims.** No user study was run (no real CSAT
  responses exist in the repo). Nothing is claimed about whether one
  mode is better understood than another.
- **Real-world generalization.** All data is synthetic; no claim about
  production banking systems, real accounts or real incidents.
- **Causal claims.** Neither the linkage chains nor the PAS results nor
  CSAT movements support causal statements; the code labels them
  association/hypothesis and the tests enforce it.
- **Statistical power.** Stage timings are single-run wall-clock
  measurements on one machine; experiments have small, deterministic
  sample sizes.
- **Novelty of individual methods.** See §1 — none of the underlying
  methods is claimed as new.

## 6. Literature review sources (consulted online, October 2026)

- Liao & Variengien, "Explanation is Not a Technical Term" (arXiv
  2207.00007) — audience/functional-role framing of explanations.
- Adaptive XAI workshop series (IUI 2024/2025) and AXAI — adaptive,
  tailored explanation interfaces as an established research theme.
- "Connecting Algorithmic Research and Usage Contexts" (HCOMP 2022) —
  contextualized XAI evaluation criteria.
- NIST AI 800-4, "Challenges to the monitoring of deployed AI systems"
  (2026) — post-deployment monitoring as a recognized need with nascent
  best practice.
- Recent agentic-traceability preprints (e.g. causal attribution for
  agentic decisions, agent stability indices) — traceability/stability
  of agent decisions is an actively researched area.

*If this document is ever produced offline, this section must be replaced
with "novelty unverified — no literature access at time of writing".*
