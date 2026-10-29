# TRACE-XAI Phase Plan

- [x] **Phase 0 — Assessment & plan**: environment/project inspection, stack decision,
      architecture proposal (approved).
- [x] **Phase 1 — Simulated banking agent**: deterministic synthetic dataset
      (`configs/bank_dataset.yaml`), banking environment with constraint engine,
      NL task parser, replaceable agents (rule-based + LangChain/Ollama),
      structured `ExecutionResult` trace records, 43 passing tests.
- [x] **Phase 2 — Structured execution traces**: `core/trace.py` schema
      (EventType/TraceEvent/ExecutionTrace/Provenance/ErrorInfo), single-writer
      `ExecutionRecorder`, rule+LLM agent integration (`run_traced`), honest nulls
      for missing evidence, conflict representation, committed sample trace,
      73 passing tests.
- [x] **Phase 3 — Q1 decision explainability**: machine-readable requirement registry
      (`configs/requirements_bank.yaml`), deterministic requirement-to-action matcher,
      per-requirement assessments (SATISFIED/VIOLATED/UNKNOWN) with evidence
      citations, conflict reporting, concise rationale; 89 passing tests.
- [x] **Phase 4 — Q2 behavior-deviation explainability**: interpretable trajectory
      anomaly detector (8 rule-based signals), hybrid explainer combining Phase 3
      requirement checking with anomalies, confirmed/possible/unknown association
      labelling (no causal claims), 11 passing tests; 111 tests total.
- [x] **Phase 5 — Q3 counterfactual explainability**: minimal alternative-action
      generation (clamp to binding constraint, verified-beneficiary switch),
      deterministic sandbox simulation against recorded pre-action state,
      controlled replay in an isolated environment clone (explicitly recorded,
      never blended with the traced execution), requirement comparison,
      improves/worsens impact labelling, uncertainty rubric; 15 passing tests;
      126 tests total.
- [x] **Phase 6 — SLM rationale layer**: `rationale/` package — structured
      prompt template (evidence packet, no CoT requests), replaceable provider
      interface (local Ollama via raw API with `think: false`, isolated
      OpenAI-compatible HTTP interface, test double), five-section
      user-facing rationale with per-claim citation grounding
      (prose + claims scanned; unresolvable ids quarantined), deterministic
      non-LLM fallback (0.36 ms); 21 unit + 4 real-model integration tests;
      151 tests total.
- [x] **Phase 7 — Q4 explanation validation**: `validate/` package — claim
      extraction (structured claims + citable sentences), evidence retrieval
      from the Phase 6 packet, deterministic verification rules (limit/balance/
      verification/outcome/requirement-status/asserted-number), lexical NLI
      backstop (can only contradict), optional LLM-as-judge (semantic claims
      only, confidence capped, never overrules rules), guardrail decisions
      PASS/FLAG/BLOCK; 20 unit + 3 real-model integration tests; 174 tests
      total.
- [x] **Phase 8 — Integrated pipeline**: `pipeline.py` — `TraceXAIPipeline`
      composes the validated Q1–Q4 components behind a single call
      (`analyze_execution(task_id)` / `run(task)`), producing a 15-section
      `FinalXAIReport` with per-stage timings and embedded sub-reports.
      Controlled-failure robustness is exercised by end-to-end scenarios
      C/D (injected anomalies), F (missing evidence), G (conflicting
      evidence), H (unsupported explanation blocked) in
      `tests/e2e/test_pipeline.py`; 12 e2e + 1 real-model integration test;
      189 tests total.

## Final phase (extension plan — see `docs/final_phase_plan.md`)

- [x] **Increment 1 — Adaptive explanation presentation**: `presentation/`
      package — TECHNICAL/BUSINESS/NORMAL/ELI5 modes generated as a pure
      deterministic function of the validated `FinalXAIReport` (statuses,
      counts, guardrail and evidence refs copied, never recomputed);
      FACT/INFERENCE/UNKNOWN labels on every line of the audit text
      format (the readable UI panels later framed fact vs guess by
      section instead — see reading_the_explanations.md); ELI5 analogies
      labelled "(not evidence)"; socket-blocking test proves no model is
      consulted; 24 tests.
- [x] **Increment 2 — Metrics**: `metrics/` package — latency split
      (agent execution / XAI analysis / end-to-end, all ms, perf_counter,
      unmeasured = UNKNOWN not 0), real provider token usage
      (`model_usage` captured from Ollama `last_usage`) with cost UNKNOWN
      unless pricing supplied (then ESTIMATED + source), decomposed claim
      quality (`unsupported = (contradicted + unknown)/total` with
      UNKNOWN never called a hallucination), CSAT NOT_COLLECTED with a
      deterministic labelled simulated variant + `CSATRecorder`; 23 tests.
- [x] **Increment 3 — Post-action stability**: `stability/` package —
      bounded monitoring window (configurable), intervention types,
      observation gaps, statuses INTERVENTION_DETECTED /
      NO_INTERVENTION_OBSERVED / INSUFFICIENT_OBSERVATION, recorded end
      reason, intervention-free duration; injected logical-clock time
      (no waiting ever); never claims correctness or indefinite
      stability; 24 tests.
- [x] **Increment 4 — Metric→trace linkage**: `linkage/` package —
      METRIC → OBSERVED CHANGE → RELEVANT TRACE EVENTS → EXPLANATION
      CLAIM → EVIDENCE REFERENCES → VALIDATION STATUS chains, labelled
      ASSOCIATION/HYPOTHESIS (never causal; CSAT/retention always
      hypothesis); field-level scoping; 12 tests.
- [x] **Increment 5 — Experiments A–E**: `experiments/
      final_phase_experiments.py` — (A) presentation consistency across
      identical traces, (B) no-validation baseline vs full pipeline with
      false-acceptance/detection metrics against synthetic ground truth,
      (C) metric evaluation over 5 scenarios, (D) PAS over 7 cases
      incl. interruption/short window, (E) Phase 8 vs extended system
      compatibility; each with documented limitations; results in
      `experiments/results/final_phase_results.json`; 13 regression tests.
- [x] **Increment 6 — Evaluator GUI**: `app.py` (Gradio 5.38) — 7 tabs /
      12 capabilities (scenario run, trace inspection, all explanations,
      mode selection, evidence & status, latency+cost, claim lists, CSAT
      controls, After-Effects, metric→trace linkage, consolidated
      report, export text+JSON); every metric shows unit + basis;
      launch verified (HTTP 200); 28 tests.
- [x] **Increment 7 — Docs & summary**: README status/run/layout,
      this phase log, `docs/innovation_summary.md` (reused vs integrated
      vs new vs demonstrated vs unverified, with an online literature
      review), final test run: **312 tests passing** (unit + e2e +
      integration incl. real-model tests).

## Open questions (tracked)

- Semantic/embedding-based matching deferred: no local embedding model (disk budget).
- LLM judge shipped with `qwen3.5:4b`; a second judge model comparison
  (e.g. `llama3.2`) remains open — judge verdicts are capped and advisory only.
- Demo UI: **resolved** — Gradio evaluator GUI shipped as `app.py`
  (FastAPI alternative not needed; Gradio 5.38 was already installed).
