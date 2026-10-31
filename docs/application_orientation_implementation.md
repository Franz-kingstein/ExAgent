# Application-Orientation & Code of Implementation

---

## A. Application-Orientation

### A.1 Domain and Application Class

TRACE-XAI is positioned in the domain of **AI-assured digital banking**, with
the concrete application being the explanation and audit of *transaction
agents* — software agents that execute customer banking requests such as
fund transfers under registered policy constraints (per-transaction limits,
beneficiary verification, balance sufficiency).

In application class, TRACE-XAI is a **post-hoc explainability and assurance
layer**: it does not replace or modify the banking agent. It wraps the agent
at its observable interface, records every action the agent takes, and
produces an evidence-grounded explanation of the agent's behaviour together
with a machine-checked verdict on whether that explanation may be published
as-is. Any agent that can be observed at the tool-call level can therefore be
placed inside it without modification.

### A.2 Intended Users and Their Application Needs

| User role | Application need met by TRACE-XAI |
|---|---|
| AI/ML engineer | Step-by-step execution trace, timings, counterfactual replay results — debugging an agent's behaviour without guessing |
| Risk & compliance officer | Requirement-by-requirement status (SATISFIED / VIOLATED / UNKNOWN) with business-impact wording; a guardrail that withholds flawed explanations |
| Auditor / regulator | Full evidence index (E1…En), per-claim verdicts with verification method and confidence, and a text/JSON export that labels every line's epistemic class |
| Operations / support | Plain-language and lay-language renderings of the same validated analysis, for explaining a declined transaction to a customer |
| Evaluator / researcher | A controlled synthetic sandbox (six scenarios) to assess explanation quality, honesty behaviour and metric integrity |

### A.3 Application Scenarios

The system ships with six synthetic application scenarios, each a deliberate
application situation rather than a random test case:

| Scenario | Application situation | What it exercises |
|---|---|---|
| A | Successful transfer within limits | Clean path; all requirements satisfied |
| B | Transfer above the per-transaction limit | Policy violation with full evidence chain |
| C | Invalid amount submitted twice | Trajectory anomaly without a policy violation |
| D | Violation **and** anomaly together | Both risk categories in one run |
| F | Missing state recording | UNKNOWN end-to-end; nothing upgraded to a verdict |
| G | Two opposite recorded results | Conflicting evidence surfaced, never silently resolved |

A custom task can also be typed into the GUI, which executes in the same
synthetic sandbox.

### A.4 Operational Context and Deployment Orientation

- **Execution environment**: entirely local — the banking sandbox, the
  evidence base, the explanation pipeline and (optionally) the small language
  model all run on the operator's machine or an on-premise server. No data
  leaves the environment by default; an external model endpoint exists as an
  explicitly configured, isolated option only.
- **Interfaces**: (i) a 7-tab Gradio GUI for human evaluators, (ii) a Python
  pipeline API (`TraceXAIPipeline.run(...)`) for embedding into larger
  systems, and (iii) text/JSON file exports as the audit artefact.
- **Integration point**: in a production banking stack, TRACE-XAI would sit
  alongside the agent's execution path as an asynchronous assurance service —
  consuming the agent's trace stream, and emitting explanations, verdicts and
  metric updates. (This build demonstrates that role against the synthetic
  sandbox; no real bank integration is claimed.)
- **Application boundary (stated honestly)**: all data in this build is
  synthetic and locally generated; the CSAT figures exist only when a person
  records a real rating during a session; simulated values are always labelled
  as such.

---

## B. Code of Implementation

### B.1 Implementation Stack and Rationale

| Layer | Technology | Reason for choice |
|---|---|---|
| Language | Python 3.11+ | Ecosystem for AI/agent tooling; typing for schema safety |
| Schemas & invariants | Pydantic v2 | Every stage's data contract is an executable model (validation + documentation in one) |
| GUI | Gradio 5 | Fast, testable web UI over pure Python handlers |
| SLM runtime | Ollama (`qwen3.5:4b`, local) | On-premise inference; no data egress; real token accounting via raw HTTP |
| Tests | pytest | Unit / integration / e2e separation; no network in unit layer |
| Versioning | Git + GitHub | Phase-wise history; reproducible releases |

### B.2 Coding Standards and Conventions

1. **Explicit types everywhere** — function signatures and Pydantic models
   carry full type information; no untyped dictionaries cross module borders.
2. **Docstrings carry the rules** — each module's docstring states the
   honesty invariants it enforces, so the code is its own specification
   (e.g. the recorder's *"null if genuinely not observed — never fabricated"*).
3. **Deterministic by construction** — explanation stages are pure functions
   of the validated report; the SLM runs at `temperature=0, seed=42`; no
   stage may recompute a status it did not originate.
4. **Record vs. re-render separation** — recorded values (statuses, counts,
   evidence) live on models; renderers (`to_text`, `to_markdown`) may only
   reformat them. Tests assert composition equality between renderers.
5. **Unknown is not zero** — unmeasured quantities are typed UNKNOWN /
   `None` with an explicit basis label; they are never defaulted to 0.
6. **Fail honestly** — every model-backed stage has a deterministic fallback
   that records *why* it fell back (provider down, unparseable output,
   quarantined citations).
7. **No network in the unit-test layer** — model/daemon paths are covered by
   integration tests with skip-when-unavailable semantics.
8. **Narrow interfaces** — replaceable components (agent, SLM provider,
   judge) implement minimal protocols (e.g. `generate(system_prompt,
   user_prompt) -> str`), so the framework is not coupled to any vendor.

### B.3 Architectural Pattern of the Code

The codebase follows a **pipeline of injected components** pattern: each
stage is an independent class composed by the orchestrator with dependency
injection (nothing is reimplemented in the orchestrator):

```python
# application / embedding code
pipeline = TraceXAIPipeline(
    agent=RuleBasedAgent(BankEnvironment.from_dataset()),
    rationale_provider=OllamaRationaleProvider(timeout=90.0),  # optional
)
result, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
```

The recorder follows an **emit-events pattern**: every observation becomes an
immutable, provenance-tagged `TraceEvent`, and all later stages read only
these events:

```python
def record_tool_invocation(self, action, tool_name,
                           tool_input=None, state_before=None) -> TraceEvent:
    """Record an observed tool invocation. ``state_before``: null if unavailable."""
    return self._emit(
        EventType.TOOL_INVOCATION, action=action, tool_name=tool_name,
        tool_input=tool_input, state_before=state_before,
        provenance=Provenance(source=ProvenanceSource.ENVIRONMENT,
                              source_id=tool_name))
```

### B.4 Module-wise Implementation

| Module (package) | Key classes | Implementation responsibility |
|---|---|---|
| `banking/` | `BankEnvironment` | Synthetic accounts, limits, beneficiaries; scenario fixtures |
| `agent/` | `RuleBasedAgent` | Deterministic policy-enforcing agent (optional LLM agent variant) |
| `core/` | `ExecutionRecorder`, `ExecutionTrace`, `TraceEvent` | Event capture, provenance, immutable trace |
| `requirements/` | requirement registry model | Registered constraints loaded from `configs/` |
| `explain/` | `RequirementMatcher`, `DecisionExplainer`, `DeviationExplainer`, `CounterfactualExplainer` | Q1–Q3 stages + evidence index derivation |
| `rationale/` | `RationaleGenerator`, `OllamaRationaleProvider`, `RationaleContext` | Evidence packet, prompts, SLM call, grounding & quarantine, deterministic fallback |
| `validate/` | `ExplanationValidator`, `LLMJudge`, `extract_claims` | Q4: claim extraction, rule/NLI verification, guardrail |
| `pipeline.py` | `TraceXAIPipeline`, `FinalXAIReport` | Orchestration, per-stage timing, 15-section report |
| `presentation/` | `present_report`, `PresentedExplanation` | 4 audience modes; story/proof renderers; invariance digests |
| `metrics/` | `collect_metrics`, `MetricBasis` | Latency, cost, claim quality, CSAT with basis labels |
| `stability/` | `PostActionStabilityMonitor` | Post-action intervention window; bounded claims |
| `linkage/` | `link_metric`, `MetricTraceLink` | Metric→trace chains (ASSOCIATION / HYPOTHESIS) |
| `app.py` | Gradio handlers | 7-tab evaluator GUI; state-chained panel refresh |

### B.5 Key Implementation Algorithms

**(i) Requirement matching (decision table).** Deterministic; missing
evidence is never assumed satisfied:

```text
no candidates        -> UNKNOWN   (missing evidence is never assumed satisfied)
all candidates OK    -> SATISFIED
any candidate fails  -> VIOLATED
candidates disagree  -> UNKNOWN, all observations listed as conflicting evidence
```

**(ii) Evidence grounding of SLM output (quarantine algorithm).** Every
citation the model emits must resolve to the supplied packet:

```python
bad_refs = [r for r in refs if not _REF_PATTERN.match(r) or r not in allowed]
if bad_refs:
    unsupported.append(f"{text[:60]}: unresolvable refs {bad_refs}")
    continue                       # quarantined — never presented as supported
if label is not ClaimLabel.UNKNOWN and not refs:
    unsupported.append(f"{text[:60]}: factual claim without citation")
    continue
```

A second pass scans the prose sections for unknown identifiers; any hit sets
`grounded=False`, which the guardrail can never override to PASS.

**(iii) Guardrail decision (publication policy).**

```python
if not rationale.grounded:      return BLOCK   # quarantined upstream
if contradicted:                return BLOCK   # any refuted claim
if rate >= 0.5:                 return BLOCK   # too little of it is evidenced
if unknown and rate > 0.0:      return FLAG    # unverifiable claims present
if not total:                   return FLAG    # nothing to validate
return PASS
```

**(iv) Counterfactual labelling.** Each counterfactual records its basis and
is rendered as `replay-confirmed prediction` (state replay performed) or
`prediction only`, with an uncertainty value — never as a fact.

**(v) Presentation invariance.** Cross-mode digests (statuses, evidence refs,
counts) are computed from the models and compared for equality; the UI
displays the digest on every run, and tests assert equality across all four
modes.

**(vi) GUI state sequencing.** Panels that read the post-run state are
chained with `.then()` so they execute after the run commits its state —
implemented after a browser-only race (parallel same-click listeners
snapshotted the pre-run state) was found and fixed with a regression test.

### B.6 Quality Assurance of the Code

- **334 automated tests** across 26 test files: per-module unit tests,
  end-to-end pipeline tests, presentation-invariant tests, GUI wiring tests
  that drive every event through Gradio's real postprocess layer, and
  integration tests for the real SLM path (skipped when the daemon is down).
- **Phase-wise version control**: each implementation phase is a reviewed
  commit (`Phase N: …`, `Final phase inc.N: …`) with its tests and docs in
  the same commit; releases are pushed to GitHub only after the full suite
  passes.
- **Self-documenting honesty**: the test-suite encodes the research rules
  (no fabricated evidence, no UNKNOWN upgrades, no causal claims, basis
  labels on every metric), so any future change that weakens honesty fails
  the build.

---

### References to source (for the appendix)

| Topic | File |
|---|---|
| Pipeline & 15-section report | `src/trace_xai/pipeline.py` |
| Evidence capture | `src/trace_xai/core/recorder.py` |
| Requirement matching rules | `src/trace_xai/explain/matcher.py` |
| SLM grounding & fallback | `src/trace_xai/rationale/generator.py` |
| SLM provider (`qwen3.5:4b`, `think: false`) | `src/trace_xai/rationale/provider.py` |
| Guardrail policy | `src/trace_xai/validate/validator.py` |
| Presentation invariants | `src/trace_xai/presentation/presenter.py` |
| Linkage labels | `src/trace_xai/linkage/linker.py` |
| Evaluator GUI | `app.py` |
| Test suite | `tests/unit`, `tests/integration`, `tests/e2e` |
