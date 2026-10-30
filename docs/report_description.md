# TRACE-XAI — Project Description (report-ready)

> Suggested citation block: *TRACE-XAI: An Evidence-Grounded Explainability and
> Assurance Framework for Agentic Banking Workflows*, B.Tech project report,
> 2026. Repository: `github.com/Franz-kingstein/ExAgent`.

---

## 1. Abstract

TRACE-XAI is a **model-agnostic explainability and assurance framework** for
banking AI agents. It wraps any agent's execution, records every step as an
immutable trace, and produces a structured, **evidence-grounded explanation**
whose factual claims are themselves verified against the recorded evidence.
The framework answers three explanation questions — *why this decision* (Q1),
*what deviated from expectation* (Q2), and *what else could have happened*
(Q3) — and adds a fourth stage, *claim validation* (Q4), which assigns each
claim in the generated explanation a verdict of SUPPORTED, CONTRADICTED or
UNKNOWN and drives a show / flag / block **guardrail**. Explanations are
re-rendered for four audiences (technical, business, general, lay) from one
validated report, with statuses, evidence references and counts provably
identical across modes. All data is synthetic and generated locally; no
external service is contacted by default, and the only machine-learning
component — an optional local small language model that drafts rationale
prose — is quarantined behind a citation-grounding layer with a deterministic
fallback. The system comprises ~12,000 lines across 49 Python modules and 26
test files (**334 automated tests**), plus a 7-tab Gradio evaluator GUI.

---

## 2. Introduction and Motivation

AI agents in banking (transfers, limits, beneficiary checks) increasingly
drive consequential decisions, yet their behaviour is typically exposed only
as logs or a single output. Regulators and users increasingly demand not just
*what* the system did but **why** — and, crucially, whether the offered
explanation is itself trustworthy. Explanations that hallucinate evidence are
worse than none at all.

Existing model-centric explainability (LIME, SHAP, attention maps) assumes a
differentiable predictor with feature vectors; it does not apply to agentic
workflows whose behaviour emerges from rules, tool calls and state changes.
TRACE-XAI therefore takes a **process-centric** stance: the object of
explanation is the *execution trace* — the recorded sequence of observations —
and the explanation is only as good as the evidence it cites. The framework's
core novelty is that **explanations are validated like software artefacts**:
claims are extracted, checked against the evidence base, verdicted, and
passed through a guardrail before they may be shown.

## 3. Objectives

1. Record every agent action as an immutable, provenance-tagged execution
   trace (the evidence base).
2. Explain decisions (Q1), detect deviations and trajectory anomalies (Q2),
   and generate counterfactual alternatives (Q3) — deterministically.
3. Generate a user-facing rationale with an optional local SLM that can
   never introduce unverifiable facts.
4. Verify every claim of the generated explanation (Q4) and gate publication
   with a guardrail (PASS / FLAG / BLOCK).
5. Present the one validated explanation to four audiences without changing
   any conclusion.
6. Report metrics honestly: every number carries its unit and its basis
   (measured / estimated / simulated / unknown).
7. Provide post-action stability monitoring and metric-to-trace linkage that
   stop at association, never causation.

---

## 4. System Architecture

```
 User task ──▶ Banking agent (rule-based; optional LangChain/LLM agent)
                     │  every step recorded
                     ▼
            Execution trace (immutable events, provenance-tagged)
                     │
     ┌───────────────┴───────────────────────────────┐
     │ 1. Requirement matching      (registry)       │──┐
     │ 2. Decision explanation      (Q1)             │  │
     │ 3. Deviation + anomaly       (Q2)             │  │  evidence
     │ 4. Counterfactual analysis   (Q3, replay)     │◀─┘  index
     └───────────────┬───────────────────────────────┘  (E1..En)
                     ▼
     5. Rationale layer — deterministic templates by default;
        optional local SLM (Ollama qwen3.5:4b) over the evidence
        packet only, with citation grounding + quarantine + fallback
                     ▼
     6. Claim validation (Q4) — deterministic rules → lexical NLI →
        optional LLM judge; SUPPORTED / CONTRADICTED / UNKNOWN
        per claim + guardrail PASS / FLAG / BLOCK
                     ▼
            FinalXAIReport (15 sections, timings, usage)
                     │
     ┌───────────────┼──────────────────────────────┐
     ▼               ▼                              ▼
 7. Presentation   8. Metrics                  9. Stability / linkage
    4 audiences,     latency/cost/claims/CSAT,   post-action intervention
    one validated    every number + basis        monitoring; metric→trace
    report           label                       chains (association only)
                     ▼
            Gradio evaluator GUI (7 tabs) + text/JSON export
```

Data-flow order is 1→6; validation runs *after* the rationale layer because
it verifies the rationale's claims. Each stage is timed; the report carries
its own performance profile.

The integrated report model (excerpt, `src/trace_xai/pipeline.py`):

```python
class FinalXAIReport(BaseModel):
    """The integrated report: 15 user-facing sections + embedded sub-reports."""
    task: str                          # 1. Task text
    requirements: list[RequirementAssessment]   # 2. registered requirements
    agent_action: list[str]            # 3. observed action + outcome
    decision_rationale: str            # 4. Q1 rationale text
    requirement_satisfaction: dict[str, str]    # 5. id -> status
    deviations: list[str]              # 6. Q2 deviation summary lines
    trajectory_anomalies: list[str]    # 7. detected anomalies
    counterfactuals: list[str]         # 8. Q3 alternative-action summaries
    evidence: list[str]                # 9. evidence index lines
    validation: ValidationReport       # 10. verdicts, counts, guardrail
    supported_claims: list[str]        # 11. SUPPORTED claim texts
    contradicted_claims: list[str]     # 12. CONTRADICTED claim texts
    unknown_claims: list[str]          # 13. UNKNOWN claim texts
    uncertainty: list[str]             # 14. confidence/uncertainty lines
    limitations: list[str]             # 15. explicit limitations
    # embedded sub-reports (audit trail; nothing is flattened away)
    decision: DecisionExplanation
    deviation: DeviationExplanation
    counterfactual_explanation: CounterfactualExplanation
    rationale: StructuredRationale
    timings_ms: dict[str, float]
    model_usage: list[TokenUsage]
```

---

## 5. Honesty-First Design Principles

The framework's distinguishing feature is a set of enforced invariants
(asserted by the test-suite, not merely intended):

| # | Invariant | Enforced by |
|---|-----------|-------------|
| 1 | Statuses, counts, guardrail and rate are **copied, never recomputed** downstream | presentation tests, digests |
| 2 | Missing evidence yields **UNKNOWN**, never SATISFIED | requirement matcher |
| 3 | Conflicting evidence is **reported, not resolved** | matcher lists all conflicting items |
| 4 | An UNKNOWN is never upgraded by wording changes or simplification | presentation invariant tests |
| 5 | Every claim carries its evidence refs; ungrounded claims are **quarantined** | rationale grounding layer |
| 6 | No hidden chain-of-thought is requested or exposed | `think: false`, prompt policy |
| 7 | Every metric states its **unit and basis**; unmeasured = UNKNOWN, not 0 | metrics collector |
| 8 | Satisfaction proxies (CSAT/retention) are never causal evidence | linkage rules, tests |
| 9 | All data synthetic; no external calls by default | safety policy, provider design |
| 10 | Presentation re-renders only — the readable panels and audit text derive from the same recorded values | composition tests |

---

## 6. Modules (with code)

### 6.1 Synthetic banking sandbox and rule-based agent (`banking/`, `agent/`)

A local, deterministic banking environment (accounts, balances, per-transaction
limits, verified beneficiaries) loaded from `configs/bank_dataset.yaml`, with
a rule-based agent enforcing registered requirements. Nothing is fetched from
the network; every run is reproducible.

### 6.2 Execution tracing and evidence capture (`core/`)

The `ExecutionRecorder` captures each tool invocation and result with its
state and provenance; `None` means *genuinely not observed* — never fabricated
(excerpt, `core/recorder.py`):

```python
def record_tool_invocation(self, action, tool_name,
                           tool_input=None, state_before=None) -> TraceEvent:
    """Record an observed tool invocation. ``state_before``: null if unavailable."""
    return self._emit(
        EventType.TOOL_INVOCATION, action=action, tool_name=tool_name,
        tool_input=tool_input, state_before=state_before,
        provenance=Provenance(source=ProvenanceSource.ENVIRONMENT,
                              source_id=tool_name))

def record_tool_result(self, action, tool_name, tool_output,
                       state_after=None, success=None, error=None,
                       related_event_id=None, conflict_with_event_id=None):
    """Record an observed tool result. ``tool_output``/``state_after``: null
    if genuinely not observed — never fabricated."""
```

From these events the **evidence index** (E1, E2, …) is derived: one line per
observed value, e.g. `E1 = TRC-SYN-0001-E002.tool_input.amount_inr = 75000`.
Every later stage cites only these lines.

### 6.3 Requirement registry and matching (`requirements/`, `explain/matcher.py`)

Requirements are registered constraints (e.g. *REQ-AMT-001: transfer ≤
₹50,000*). Matching is deterministic; its decision rules are stated in the
module itself:

```text
- no candidates      -> UNKNOWN (missing evidence is never assumed satisfied)
- all satisfied      -> SATISFIED
- any violated       -> VIOLATED
- disagreeing statuses -> UNKNOWN with all evidence listed as conflicting
```

Conflicts are surfaced, never silently resolved:

```python
conflicting = [e for o in observations for e in o.evidence]
...
RequirementAssessment(..., status=RequirementStatus.UNKNOWN,
                      conflicting_evidence=conflicting)
```

### 6.4 Explanation stages Q1–Q3 (`explain/`)

- **Q1 decision explainer** — links the observed outcome to the requirement
  assessments and evidence that entail it.
- **Q2 deviation + anomaly detectors** — trajectory statistics against the
  expected step sequence; the relation between anomalies and violations is
  explicitly reported as *"none (never causal)"* when they merely co-occur.
- **Q3 counterfactual explainer** — replays the recorded state under a
  modified action in a deterministic simulator. Each counterfactual carries
  its basis and is labelled *replay-confirmed prediction* or *prediction
  only*, with an uncertainty value:

```text
CF-0001: amount reduced from 75000 to the per-transaction limit 50000;
predicted outcome: success (basis deterministic_simulation);
replay performed: True (txn TXN-SYN-0001, OK); impact: improves;
uncertainty: 0.05 [replay-confirmed prediction]
```

### 6.5 Rationale layer with optional local SLM (`rationale/`)

The rationale layer *communicates* what Q1–Q3 established; it never adds
facts. Pipeline (module docstring, `rationale/generator.py`):

```text
1. build_context   — deterministic evidence packet from trace + Q1/Q2/Q3.
2. render_prompt   — structured system/user prompts (no CoT requests).
3. provider call   — optional; any RationaleProvider; failures contained.
4. parse + ground  — strict JSON parse; every claim citation must resolve
                     to [ALLOWED CITATION REFS]; ungrounded claims are
                     quarantined (never presented as supported).
5. deterministic fallback — if no provider or unusable output, the five
                     sections are composed from the packet (llm_used=False).
```

Grounding excerpt — a factual claim without a resolvable citation is
quarantined, and explicitly-UNKNOWN claims need no citation:

```python
bad_refs = [r for r in refs if not _REF_PATTERN.match(r) or r not in allowed]
if label is ClaimLabel.UNKNOWN and not refs:
    grounded.append(RationaleClaim(text=text, label=label, evidence_refs=[]))
    continue
if bad_refs:
    unsupported.append(f"{text[:60]}: unresolvable refs {bad_refs}")
    continue
if label is not ClaimLabel.UNKNOWN and not refs:
    unsupported.append(f"{text[:60]}: factual claim without citation")
    continue
```

A prose-level hallucination net additionally scans the five output sections
for identifiers that do not resolve to the packet; any hit sets
`grounded=False`, which the guardrail can never override to PASS.

### 6.6 The SLM (`rationale/provider.py`)

Which model: **Qwen3.5 (4B parameters) running locally on Ollama**
(`qwen3.5:4b`, ~3.3 GB, `localhost:11434`) — a *small* language model, chosen
because banking data must never leave the machine and because a 4B model is
sufficient for wording, not reasoning. The provider speaks the raw HTTP chat
API (no framework dependency) and captures real token usage for the metrics
layer:

```python
class OllamaRationaleProvider:
    """Local Ollama model via the raw HTTP API (`/api/chat`). ..."""
    def __init__(self, model: str = "qwen3.5:4b", temperature: float = 0,
                 seed: int = 42, base_url: str = "http://localhost:11434",
                 timeout: float = 180.0, num_predict: int = 2048) -> None:
        ...
        self._options = {"temperature": temperature, "seed": seed,
                         "num_predict": num_predict}

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model": self.model, "messages": [...],
            "stream": False, "think": False,       # no hidden reasoning block
            "options": self._options}).encode("utf-8")
        ...
        self.last_usage = {"eval_count": body.get("eval_count"),
                           "prompt_eval_count": body.get("prompt_eval_count"), ...}
```

`think: False` deserves emphasis: it both prevents the model from consuming
its output budget on a hidden reasoning block and complies with the project's
**no-hidden-chain-of-thought** rule — the model works purely from the supplied
evidence packet. `temperature=0, seed=42` make runs near-deterministic. The
GUI wires this behind an opt-in checkbox with a 90-second timeout (cold model
load); on failure the deterministic template rationale is used and the metrics
report 0 tokens — never a fabricated count.

### 6.7 Claim validation and guardrail (`validate/`)

Validation is *deterministic first and last*: an optional LLM judge only sees
claims every deterministic method left UNKNOWN, its confidence is capped, and
without evidence it cannot confer support (pipeline docstring,
`validate/validator.py`):

```text
generated explanation
        |  claim extraction       (claims array + citable sentences)
        |  evidence retrieval     (E-numbers, REQ/ANOM/CF lines)
        |  claim verification     deterministic rules -> lexical NLI -> LLM judge
        v
SUPPORTED / CONTRADICTED / UNKNOWN  ->  guardrail PASS / FLAG / BLOCK
```

The guardrail policy (excerpt, `validator.py`):

```python
def _guardrail(self, contradicted, unknown, total, rate, rationale, notes):
    """Strict policy: false facts block; uncertainty flags; clean passes."""
    if not rationale.grounded:
        ...  # upstream quarantined citations; cannot PASS
        return GuardrailDecision.BLOCK
    if contradicted:
        return GuardrailDecision.BLOCK          # any false fact blocks
    if total and rate >= self._block_rate:      # default 0.5
        ...
        return GuardrailDecision.BLOCK
    if unknown and rate > self._flag_rate:      # any unverifiable claim
        return GuardrailDecision.FLAG
    if not total:
        ...
        return GuardrailDecision.FLAG           # nothing to validate
    return GuardrailDecision.PASS
```

The **unsupported-claim rate** is `(contradicted + unknown) / total` — claims
the records could not back out of all claims checked — and feeds a documented
simulated quality score.

### 6.8 Audience-adaptive presentation (`presentation/`)

One validated report is re-rendered for **TECHNICAL / BUSINESS / NORMAL /
ELI5** audiences as a pure function — no model call, no recomputation.
Module invariants (excerpt, `presentation/presenter.py`):

```text
1. Verification statuses, claim texts, counts, guardrail and unsupported-
   claim rate are copied from the validated report, never recomputed.
2. Every mode carries the full evidence index and per-claim evidence lines.
3. Every mode carries the recorded rationale/deviation text verbatim, so
   simplifying wording never drops source facts.
4. Each presented line records its kind FACT / INFERENCE / UNKNOWN: the
   audit renderings (to_text(), JSON export) print the label on every line;
   the readable markdown panels frame fact vs guess by section
   (counterfactuals announced as predictions, analogies marked "(not
   evidence)") instead of cluttering each line with bracket-tags.
5. No model is called while re-wording (deterministic pure function).
```

The UI rendering is split into a **story half** (what happened, why, rules as
a table) and a **proof half** (claim checks, validation numbers, evidence
index, uncertainty) that compose back into one document — asserted by test:

```python
def to_markdown(self) -> str:
    return self.story_markdown() + "\n\n" + self.proof_markdown()
```

An invariance digest is displayed on every run, proving the four modes carry
identical statuses, evidence refs and counts.

### 6.9 Metrics with basis labels (`metrics/`)

Every number carries unit and basis; unmeasured values are UNKNOWN, never 0
(excerpt, `metrics/collector.py`): agent execution time is `MEASURED` only
when actually timed, otherwise the end-to-end basis is set to
`MetricBasis.UNKNOWN`; cost is computed only from provider-reported token
usage (basis `MEASURED`) and only estimated (`ESTIMATED`) when the evaluator
supplies pricing; CSAT reports `NOT_COLLECTED` until a person records a real
rating, and any simulated score is a separate, explicitly labelled field:

```python
csat_metrics = simulated_csat(report)   # labelled SIMULATED, formula shown
# real ratings: status="NOT_COLLECTED" with 0 responses until recorded
```

### 6.10 Post-action stability monitoring (`stability/`)

After an action, within a declared window, the monitor evaluates whether an
intervention occurred (human override, reversal) — and, crucially, **never
claims the action was correct**; it reports what was observed within the
window. Observation gaps (e.g. a collector offline) force an
INSUFFICIENT-evidence verdict rather than a clean bill of health.

### 6.11 Metric-to-trace linkage (`linkage/`)

When a monitoring metric moves, the linker walks an explicit chain to the
evidence an auditor should inspect — and labels the relationship honestly
(module docstring, `linkage/linker.py`):

```text
METRIC -> OBSERVED CHANGE -> RELEVANT TRACE EVENTS
       -> EXPLANATION CLAIM -> EVIDENCE REFERENCES -> VALIDATION STATUS

- ASSOCIATION — the metric movement and the linked claims live in the same
  validated run and share trace events. Co-location, not causation.
- HYPOTHESIS — no trace evidence connects the movement (unscoped metric,
  or a satisfaction metric, which is never evidence about factual claims).
```

CSAT/retention movements are therefore *always* HYPOTHESIS: a satisfaction
proxy moving is not evidence that any explanation claim is right or wrong.

### 6.12 Gradio evaluator GUI (`app.py`)

Seven tabs: (1) run a scenario (6 synthetic scenarios + custom task, optional
SLM toggle); (2) explanations in 4 audience modes with story/proof panels;
(3) evidence index and claim verdicts; (4) metrics with basis labels;
(5) After-Effects monitoring with example inputs; (6) metric→trace linkage;
(7) consolidated report + text/JSON export. State-driven panels are chained
to run *after* the run commits state — a real race condition found in the
browser and fixed by sequencing the events:

```python
run_event = run_btn.click(run_scenario, inputs=[...],
                          outputs=[run_out, trace_rows, state, expl_out,
                                   expl_proof, expl_digest, pas_out])
# panels reading the FRESH state must run after it is committed:
run_event.then(evidence_view, inputs=[state],
               outputs=[ev_head, ev_index, claims])
run_event.then(metrics_view, inputs=[state, use_pricing, ...],
               outputs=[metrics_out])
run_event.then(consolidated_view, inputs=[state], outputs=[report_out])
```

---

## 7. Innovations

1. **Validated explanations** — explanations are treated as software
   artefacts: claims extracted → verified → verdicted → guarded.
2. **Guardrail for explanations** (PASS/FLAG/BLOCK) as an explicit,
   test-pinned publication policy.
3. **Quarantined SLM** — the local SLM drafts prose only; every citation it
   emits must resolve to the evidence packet or it is quarantined and the
   explanation is flagged ungrounded.
4. **Deterministic floor everywhere** — every model-backed stage has a
   non-LLM fallback (templates, rules, replays); the system degrades
   honestly, never fails silently.
5. **UNKNOWN as a first-class outcome** — missing evidence produces UNKNOWN
   end-to-end (requirements, claims, metrics, stability), never a guess.
6. **Conflict reporting over resolution** — disagreeing observations are
   listed, not silently reconciled.
7. **Counterfactuals with provenance** — replay-confirmed vs prediction-only
   labels with uncertainty values.
8. **Four-mode presentation with invariance digests** — audience adaptation
   provably cannot change conclusions.
9. **Basis-labelled metrics** — measured/estimated/simulated/unknown on every
   number, including simulated CSAT with its formula shown.
10. **Association-only linkage** — a monitoring-to-evidence bridge that stops
    honestly at association/hypothesis.
11. **No-hidden-CoT compliance** — the SLM runs with thinking disabled and is
    never asked for reasoning traces; only evidence-grounded output is kept.
12. **Full synthetic locality** — no data leaves the machine; external
    providers exist only as an explicitly configured, isolated option.

---

## 8. Evaluation

**Scenarios (all synthetic):** A successful transfer · B limit violation ·
C trajectory anomaly (invalid amount twice) · D violation + anomaly ·
F missing evidence (no state recorded) · G conflicting evidence (two opposite
results).

**Experiments** (final-phase experiment suite, run in CI):
- *A — Adaptive presentation consistency*: the same validated report is
  presented in all four modes; status digests, evidence-ref digests and
  counts are compared for equality (mode_views = 4 × traces).
- *B — SLM rationale honesty*: with no provider configured, the rationale
  layer must fall back deterministically with zero token cost — the honest
  default path, not an error.

**Testing:** 334 automated tests across 26 files — unit tests per module,
end-to-end pipeline tests, GUI wiring tests that drive every event through
Gradio's real postprocess layer, and integration tests that exercise the
real local SLM path when the daemon is available (skipped otherwise).
Unit tests never make network/model calls.

**Representative results (scenario B, verified live):** 3 requirements
matched (1 VIOLATED, 2 SATISFIED, 0 UNKNOWN); 4 claims checked, all
SUPPORTED; unsupported-claim rate 0.000; guardrail PASS; 6 evidence index
lines; counterfactual replay-confirmed (reduce amount → success); latency in
double-digit milliseconds; model cost reported as 0 tokens / unknown USD with
the deterministic fallback.

---

## 9. Limitations

1. **Synthetic domain only** — banking data is generated locally; no real
   ledger history, so controlled replays cannot reconstruct it (stated in
   every report's limitations section).
2. **Rule-based reference agent** — the framework is model-agnostic and an
   optional LangChain/LLM agent path exists, but the demonstrated runs use a
   deterministic agent.
3. **SLM prose is quarantined, not perfected** — an imaginative model can
   still waste the 90-second budget or produce quarantined claims; the system
   responds by falling back and flagging, which is honest but plainer.
4. **CSAT is not a user study** — real satisfaction exists only when a
   person records a rating in the session; the simulated score is labelled
   demonstration data.
5. **Stability verdicts depend on observation inputs** — intervention and
   gap data are supplied by the evaluator and are explicitly simulated.
6. **Association ceiling** — linkage deliberately makes no causal claim;
   establishing causation would require controlled studies beyond scope.

## 10. Conclusion and Future Scope

TRACE-XAI demonstrates that an agent's behaviour can be explained *and* the
explanation itself assured: every statement resolves to a recorded event,
every uncertainty is preserved, and publication is gated by a guardrail.
Future work: adapters for real bank datasets, an LLM-agent evaluation
campaign using the same trace schema, a user study to replace the simulated
CSAT, persistence for longitudinal After-Effects monitoring, and
tighter SLM integration (constrained decoding to the citation grammar).

---

## 11. Technology Stack and Repository Layout

- **Language**: Python 3.11+ · **Models**: Pydantic v2 (schemas +
  invariants) · **GUI**: Gradio 5 · **SLM runtime**: Ollama (qwen3.5:4b,
  local) · **Tests**: pytest · **Versioning**: git/GitHub.
- Project start: 17 October 2026 (first commit); core phases completed
  17–19 October 2026.

```
configs/            bank + requirement registries, scenarios, judge prompts
src/trace_xai/
  banking/          synthetic environment (state, dataset, results)
  agent/            parser, rule-based agent, optional LangChain agent
  core/             trace schema + execution recorder (evidence capture)
  requirements/     requirement model + registry
  explain/          matcher (Q1), decision (Q1), deviation+anomaly (Q2),
                    counterfactual (Q3)
  rationale/        context packet, prompts, template fallback, SLM
                    providers, generator with grounding/quarantine
  validate/         claim extraction, rules, optional judge, validator,
                    guardrail
  pipeline.py       TraceXAIPipeline -> FinalXAIReport (15 sections)
  presentation/     4 audience modes; story/proof renderers; invariants
  metrics/          latency, cost, claim quality, CSAT (basis labels)
  stability/        post-action stability monitor (bounded claims)
  linkage/          metric->trace chains (association/hypothesis)
  llm/              shared low-level HTTP helpers
app.py              Gradio evaluator GUI (7 tabs)
tests/              unit + integration + e2e (334 tests)
docs/               architecture, safety, phases, innovation summary,
                    reading_the_explanations.md
```

### How to run

```bash
python -m venv .venv && ./.venv/bin/pip install -e .[dev]
./.venv/bin/python -m pytest                 # full suite (334 tests)
./.venv/bin/python app.py                    # GUI on http://127.0.0.1:7860
ollama serve && ollama pull qwen3.5:4b       # optional, for the SLM toggle
```
