# TRACE-XAI Safety & Research Policy

This document is binding for every component, dataset, experiment, and generated
artifact in this repository.

## 1. Synthetic data only (banking sandbox)

- The project uses a **simulated banking agent** with **completely synthetic/fake
  financial data**.
- **Forbidden, without exception:**
  - Real banking credentials, API keys, or tokens of any real institution
  - Real financial accounts, balances, transactions, or card numbers
  - Real personal financial data or PII of any person
- All generated data must be produced by the deterministic `faker`-based generator
  under `src/trace_xai/agent/data/` and carry a provenance flag `synthetic=True`.
- Any evidence record whose provenance is not marked synthetic must be rejected by
  validation code paths (`Provenance.synthetic` defaults to `True` and is checked in
  tests).
- Account/transaction identifiers must be generated (e.g., `SYN-` prefix), never
  resembling real instrument numbers.

## 2. No hidden chain-of-thought

- TRACE-XAI **must not** expose, request, claim access to, or simulate access to an
  agent's hidden internal reasoning (chain-of-thought).
- Explanations are built exclusively from **observable artifacts**:
  inputs, requirements, chosen actions, tool calls, tool results, state changes, and
  outcomes.
- The user-facing output is an explicit **decision rationale**, not a transcript of
  internal reasoning.

## 3. No invented evidence

- Every explanation is a list of `Claim` objects. Each claim carries:
  - a label: `FACT` | `EVIDENCE` | `INFERENCE` | `UNKNOWN`
  - `evidence_refs`: IDs of records in the append-only `EvidenceStore`
  - `method`: which component produced it
  - optional `confidence`
- `FACT` and `EVIDENCE` claims **must** reference at least one evidence record, and
  the verifier must confirm those records exist (content-hash checked).
- Claims produced by heuristics or LLM judges must be labelled `INFERENCE`.
- Where information is missing or ambiguous, the claim must be `UNKNOWN` — silence
  preferred over speculation.

## 4. LLM usage

- Default backend is **local Ollama** (no data leaves the machine). All functionality
  must also run with **LLM disabled** (deterministic mode).
- LLM outputs are advisory: they may only produce `INFERENCE`/`UNKNOWN` claims and are
  themselves subject to verification guardrails (length bounds, evidence-ref
  whitelisting, no-credential-pattern regexes).

## 5. Enforcement

- Unit tests in `tests/unit/` enforce label rules, provenance flags, and the
  no-invented-evidence invariants.
- Any change weakening these guarantees requires updating this document in the same
  commit.

## 6. GUI and exports (final phase)

- The evaluator GUI (`app.py`, Gradio) and all exports run entirely
  locally: scenarios are generated in the sandbox, no external service is
  called, and no credentials or real account data exist anywhere in the
  app (see the module docstring in `app.py`).
- **CSAT**: real ratings appear only when a person explicitly records one
  in the session (`CSATRecorder.record`); nothing is inferred or
  fabricated. The simulated CSAT/retention variant is labelled
  `SIMULATED` with its formula shown, and is never presented as user
  feedback.
- **After-Effects (stability)**: observation inputs (interventions, gaps)
  are entered/labelled as simulated; verdicts are bounded by the chosen
  window and never claimed as proof of correctness.
- **Exports** (`export_files` in `app.py`) contain only what the pipeline
  already produced: the 15-section report, the four presentations, the
  metrics bundle — all synthetic by construction.
