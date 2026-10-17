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
- [ ] **Phase 7 — Q4 explanation validation** *(next)*: evidence store +
      content hashing, rule verifier, guardrails, LLM/SLM judge, golden
      fixtures.
- [ ] **Phase 8 — Controlled failures & robustness** *(optional)*: injected
      anomalies for the deviation detector to explain.

## Open questions (tracked)

- Semantic/embedding-based matching deferred: no local embedding model (disk budget).
- Judge model choice: `qwen3.5:4b` default; compare against `llama3.2` in Phase 7.
- Demo UI (FastAPI/Gradio) decision deferred until after Phase 7.
