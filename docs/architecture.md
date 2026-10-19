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

## Execution-trace layer (Phase 2, delivered)

```
core/
  trace.py      EventType, TraceEvent, ExecutionTrace, Provenance, ErrorInfo
  recorder.py   ExecutionRecorder (single writer; ordering + IDs + logical clock)
```

- **Observable only**: no chain-of-thought, no fake "reasoning" fields.
- **Honest nulls**: unobserved tool outputs, state, requirements, and expected
  outcomes are JSON null — never defaults or placeholders.
- **Conflicts representable**: multiple `TOOL_RESULT`s may link to one
  `TOOL_INVOCATION`; contradicting results keep `conflict_with_event_id`.
- **Deterministic**: logical-clock timestamps, `TRC-*-Ennn` ids; identical runs
  produce byte-identical JSON.
- Rule-agent traces capture `state_before`/`state_after` summaries via
  `BankEnvironment.summary()`; LLM-agent traces honestly record state as null
  (not observable through the LangChain tool loop).
- Both agents expose `run_traced(task) -> (ExecutionResult, ExecutionTrace)`;
  `run(task)` is unchanged.
- Committed artifact: `data/traces/sample_transfer_success.json` — usable by
  TRACE-XAI without re-running any agent.

## Decision explainability (Phase 3, delivered)

```
requirements/model.py   ConstraintType (closed set), Requirement, load_requirements
explain/matcher.py      fact extraction + deterministic candidate observation/merge
explain/decision.py     DecisionExplainer -> DecisionExplanation (+ rationale)
explain/models.py       RequirementStatus, EvidenceItem, RequirementAssessment
configs/requirements_bank.yaml   machine-readable requirement registry
```

- Algorithm: extract observable facts from the trace → per requirement, collect
  candidate observations (deterministic comparison when operands are in
  evidence; environment `reason_code` as authoritative outcome evidence) →
  merge: no candidates ⇒ `UNKNOWN`; disagreeing ⇒ `UNKNOWN` + conflict list;
  agreement ⇒ status with confidence 0.95 (deterministic) or 0.9 (outcome).
- Every `EvidenceItem` is `EVENT_ID.DOTTED.FIELD=VALUE` — auditors can resolve
  each citation against the trace (enforced by test).
- Missing evidence is never assumed satisfied; conflicts are never silently
  resolved. No LLM anywhere in this path (`llm_used=False` by design).
- `BankEnvironment.summary()` exposes `per_txn_limits` and
  `beneficiaries_verified` so limits/verification are *trace evidence*, not
  hidden knowledge.

## Behavior-deviation explainability (Phase 4, delivered)

```
explain/anomaly.py   TrajectoryAnomalyDetector (interpretable rules, no ML)
explain/deviation.py DeviationExplainer -> DeviationExplanation (hybrid Q2)
```

- Hybrid: Phase 3 requirement assessments (reused, not duplicated) + rule-based
  trajectory anomalies: UNEXPECTED_TOOL_CALL, UNEXPECTED_EVENT_ORDER,
  MISSING_EXPECTED_EVENT, CONFLICTING_TOOL_RESULTS, REPEATED_FAILED_ACTION
  (>=2 identical failures), EXCESSIVE_RETRY (>=3), INCONSISTENT_STATE_TRANSITION,
  ACTION_INCONSISTENT_WITH_STATE.
- Category answer to "what went wrong": NONE / REQUIREMENT_VIOLATION /
  TRAJECTORY_ANOMALY / BOTH / INSUFFICIENT_EVIDENCE.
- Association between anomalies and violations is labelled
  `confirmed` (same tool call) / `possible` / `unknown` / `none` — explicitly
  **never causal** ("No causal relationship is claimed").
- Missing state (e.g. LLM traces) skips state checks instead of flagging:
  absence of evidence is not evidence of anomaly.

## Counterfactual explainability (Phase 5, delivered)

```
explain/counterfactual.py
  SandboxSimulator           deterministic what-if evaluation (no execution)
  ControlledReplayEngine     executes the alternative in an isolated clone
  CounterfactualExplainer    -> CounterfactualExplanation (Q3)
explain/models.py            ActionSpec, PredictedOutcome, ReplayRecord,
                             RequirementComparisonItem, Counterfactual
banking/environment.py       restore_summary() for replay state reconstruction
```

- Two strictly separated levels: **simulation** (deterministic prediction
  against the *recorded* pre-action state, `PredictedOutcome.basis =
  deterministic_simulation`) and **controlled replay** (the alternative is
  actually executed in a cloned `BankEnvironment` restored via
  `restore_summary(state_before)`; result recorded in `ReplayRecord`). A
  simulated alternative is never described as executed; when replay is
  disabled/unavailable, `replay=None` and the explanation says "Not replayed:
  prediction only, no execution claim is made."
- Minimal-change generation only: clamp the amount to the binding constraint
  (`min(per-transaction limit, available balance)`), or switch an unverified
  beneficiary to a verified linked one. Unrelated alternatives are never
  invented; externally supplied alternatives are evaluated the same way and
  marked `source=provided`.
- Requirement comparison reuses Phase 3 assessments of the actual trace
  (never recomputed/duplicated) against the simulated alternative statuses;
  `impact` labels improves / worsens / unchanged / unknown from the recorded
  vs. predicted outcome.
- Deterministic uncertainty rubric: base 0.05, +0.30 if any compared
  requirement is UNKNOWN, +0.10 if not validated by replay, +0.10 if
  beneficiary linkage was not observable (clamped to 1.0).
- Missing state ⇒ no minimal alternatives are generated and provided ones
  evaluate to UNKNOWN — evidence gaps are surfaced, never filled in.

## SLM rationale layer (Phase 6, delivered)

```
rationale/context.py     RationaleContext — explicit evidence packet (Q1-Q3)
rationale/template.py    structured system/user prompts (never asks for CoT)
rationale/provider.py    RationaleProvider protocol
                         ├── OllamaRationaleProvider  (local, think=false)
                         ├── OpenAICompatProvider     (isolated HTTP, opt-in)
                         └── EchoRationaleProvider    (test double)
rationale/generator.py   RationaleGenerator -> StructuredRationale (five sections)
rationale/models.py      ClaimLabel FACT|EVIDENCE|INFERENCE|UNKNOWN, RationaleClaim
```

- Research rule honored: the model is **not** asked to reveal reasoning. It
  receives a pre-computed evidence packet (requirements, action, matching
  results, deviations, anomalies, counterfactuals, evidence index, uncertainty,
  provenance) and writes a user-facing communication — nothing to infer:
  missing facts are rendered as explicit `UNKNOWN (not recorded)` lines.
- Provider replaceability: the generator only sees
  `generate(system, user) -> str`. Local Ollama is the default (raw HTTP API so
  `think: false` is enforced — qwen3.5:4b otherwise exhausts its output budget
  on hidden reasoning); external OpenAI-compatible endpoints sit behind one
  isolated, opt-in class; with no provider the same five sections are composed
  deterministically from the packet (`llm_used=False`, 0.36 ms).
- Grounding, not judgment: every cited id in the model's claims **and in the
  five section texts** must resolve to the packet (`E*`, `REQ-*`, `ANOM-*`,
  `CF-*`). Unresolvable citations are quarantined in `unsupported_claims` and
  the rationale is flagged `grounded=False` — visible, never silently trusted.
  Semantic scoring of prose is Phase 7's judge (not implemented).
- Output: five sections (decision rationale, deviation explanation,
  counterfactual summary, evidence summary, uncertainty statement) + labelled
  claims with evidence refs; `provenance.explain_layers_used_llm=False`
  documents that Q1-Q3 themselves remain deterministic.

## Explanation validation (Phase 7, delivered)

```
validate/extract.py    claim extraction (claims array + citable sentences)
validate/rules.py      deterministic rules + lexical NLI backstop
validate/judge.py      LLM-as-judge (semantic claims only, optional)
validate/validator.py  ExplanationValidator -> ValidationReport + guardrail
validate/models.py     SUPPORTED/CONTRADICTED/UNKNOWN, PASS/FLAG/BLOCK
```

- Pipeline: claim extraction → evidence retrieval (from the Phase 6 packet)
  → claim verification → per-claim verdict → guardrail decision.
- Verification ladder, deterministic first:
  1. **Deterministic rules** decide structured facts directly from cited
     evidence operands: limit relation, balance relation, beneficiary
     verification (missing value ⇒ UNKNOWN, never SUPPORTED), outcome +
     reason-code equality (conflicting success values ⇒ UNKNOWN), recorded
     requirement status, asserted-number equality.
  2. **Lexical NLI backstop** — can only *contradict* (negation/antonym
     clash), never supports.
  3. **LLM-as-judge** — only for claims every deterministic method left
     UNKNOWN; sees claim + verbatim cited evidence only; confidence capped at
     0.75; SUPPORTED with zero evidence is forced back to UNKNOWN; never
     overrules a deterministic verdict.
- Guardrails: unresolvable/absent citations ⇒ UNKNOWN (never SUPPORTED);
  any contradicted claim ⇒ BLOCK; unsupported-claim rate ≥ 0.5 ⇒ BLOCK;
  unknowns ⇒ FLAG; Phase 6 `grounded=False` input can never PASS.
- The validator judges, it never rewrites: reports list per-claim
  evidence/method/confidence plus counts, unsupported-claim rate, overall
  status, and the guardrail decision.

## Phase plan

See `docs/phases.md`.
