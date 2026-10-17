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
- [ ] **Phase 3 — Evidence store + Q1 decision explainability** *(recommended next)*:
      `EvidenceStore` (SHA-256 content hashes over trace events), Claim labels
      FACT/EVIDENCE/INFERENCE/UNKNOWN, requirement registry
      (`configs/requirements_bank.yaml`), requirement-to-action matching, first
      evidence-grounded decision explanations.
- [ ] **Phase 3 — Q2 Deviation**: hybrid requirement checking (hard/soft rules) +
      trajectory anomaly detection.
- [ ] **Phase 4 — Q3 Counterfactual**: alternative-action enumeration, comparison,
      controlled replay in the sandbox.
- [ ] **Phase 5 — Q4 Validation + experiments**: rule verifier, guardrails,
      LLM/SLM judge (local Ollama), golden fixtures, experiment notebooks.
- [ ] **Phase 6 — Controlled failures & robustness**: injected anomalies for the
      deviation detector to explain (planned, not started).

## Open questions (tracked)

- Semantic/embedding-based matching deferred: no local embedding model (disk budget).
- Judge model choice: `qwen3.5:4b` default; compare against `llama3.2` in Phase 5.
- Demo UI (FastAPI/Gradio) decision deferred until Phase 5.
