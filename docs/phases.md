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

## Open questions (tracked)

- Semantic/embedding-based matching deferred: no local embedding model (disk budget).
- LLM judge shipped with `qwen3.5:4b`; a second judge model comparison
  (e.g. `llama3.2`) remains open — judge verdicts are capped and advisory only.
- Demo UI (FastAPI/Gradio) decision deferred until after Phase 8.
