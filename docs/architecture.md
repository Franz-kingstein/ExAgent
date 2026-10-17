# TRACE-XAI Architecture

## Design: one evidence spine, four explainers

```
observable inputs ─┐
requirements ──────┼─▶ EvidenceStore (append-only, IDs + SHA-256 content hashes)
tool results ──────┤            │
actions/outcomes ──┘            ▼
                        ClaimBuilder
                              │
                              ▼
        Explanation = [ Claim{label: FACT|EVIDENCE|INFERENCE|UNKNOWN,
                              statement, evidence_refs, method, confidence} ]
                              │
        ┌─────────────────────┼──────────────────────┬─────────────────────┐
   Q1 Decision           Q2 Deviation           Q3 Counterfactual     Q4 Validation
  requirement-to-action  hybrid requirement     alternative-action    rule verifier →
   matching (Q1)         check + trajectory     comparison +          guardrails →
                          anomaly detection     controlled replay     LLM/SLM judge
```

## Invariants

1. **Append-only evidence.** `EvidenceStore.add()` never mutates or deletes; every
   record has a SHA-256 content hash; `snapshot_hash()` gives a tamper-evident digest.
2. **Claims must resolve.** A `FACT`/`EVIDENCE` claim is invalid unless every
   `evidence_refs` entry resolves to an existing record (`InventedEvidenceError`).
3. **Labels are explicit.** `FACT` (verbatim from the record) / `EVIDENCE`
   (supporting observation) / `INFERENCE` (derived, e.g. heuristic or LLM judge) /
   `UNKNOWN` (explicitly missing information).
4. **LLM-optional.** Every component runs deterministically with the LLM disabled;
   local Ollama (`qwen3.5:4b`) is the default judge/assist backend when enabled.
5. **Synthetic provenance.** All evidence originates from the simulated banking
   environment (`Provenance.synthetic=True`), see `docs/safety.md`.

## Banking layer (Phase 1, delivered)

```
src/trace_xai/
  banking/                        # environment + state (replaceable)
    state.py        BankState, Account, Beneficiary, Transaction   (data layer)
    dataset.py      YAML -> BankState, refuses non-synthetic data  (data layer)
    environment.py  BankEnvironment + BankEnvironmentProtocol       (tools/rules)
    results.py      ExecutionResult, ToolCallRecord, ReasonCode    (result layer)
  agent/                          # agents (replaceable)
    parser.py       NL task -> typed Intent (deterministic)
    rule_agent.py   RuleBasedAgent (default; deterministic)
    langchain_tools.py  LangChain StructuredTool bindings over the env
    langchain_agent.py  LangChainBankingAgent (AgentExecutor + local Ollama)
    base.py         BankingAgentProtocol
```

- **All constraints live in the environment** (active account, positive amount,
  beneficiary exists/linked/verified, per-transaction limit, sufficient balance).
  Agents cannot bypass them — the LLM only selects tools.
- **Determinism.** Rule-based agent is byte-reproducible. Ledger IDs
  (`TXN-SYN-nnnn`) and timestamps come from a logical clock, never wall time.
- **Rejected transfers are recorded** in the ledger with `status=REJECTED` and the
  reason code; `receipt_id` is set only for committed transfers.
- Every tool returns a uniform dict `{"success", "reason_code", ...}` — a ready
  evidence shape for later TRACE-XAI phases.

### Environment workarounds (this machine)

- `USE_TORCH=0` is set before LangChain imports (shared site-packages contains a
  CUDA-only torch that crashes `transformers` import). See `tests/conftest.py`.
- `langchain-core` is pinned to `<0.4` **inside the project venv** (global install
  ships an incompatible 1.x for `langchain 0.3`). Global site-packages untouched.

## Spine types (Phase 1, `src/trace_xai/core/`)

- `Provenance` — where a record came from + synthetic flag
- `Evidence` — immutable record: `id`, `kind`, `payload`, `content_hash`, `provenance`
- `EvidenceStore` — add / get / resolve / verify / snapshot
- `Claim` — labelled, evidence-referenced statement
- `Explanation` — Q1–Q4 question, subject, claims, generator, `llm_used`
- `Event` / `Step` / `Trajectory` — minimal trajectory spine (refined in Phase 2)

## Phase plan

See `docs/phases.md`.
