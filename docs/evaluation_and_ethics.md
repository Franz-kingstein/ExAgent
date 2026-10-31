# Evaluation & Methodological Insights
## (Assessing effectiveness of explainability methods)

---

## C. Evaluation of TRACE-XAI

### C.1 What "effectiveness" means for an explainability method

Explainability methods are usually assessed on *prediction-data* benchmarks
(e.g. faithfulness of feature attributions against a model's internals).
TRACE-XAI explains an agentic *process*, not a predictor, so effectiveness
is decomposed into five measurable dimensions, each with a concrete
indicator and a built-in mechanism:

| Dimension | Question it answers | Indicator | Mechanism |
|---|---|---|---|
| **Groundedness** | Is every statement backed by a recorded fact? | Evidence refs per claim; unsupported-claim rate `(contradicted+unknown)/total`; quarantined claims count | Citation grounding in the rationale layer; evidence index E1…En |
| **Faithfulness** | Do the explanation and the recorded behaviour agree? | Per-claim verdicts SUPPORTED / CONTRADICTED / UNKNOWN + verification method and confidence | Claim validation (deterministic rules → lexical NLI → optional LLM judge) |
| **Integrity under honesty pressure** | Does the method degrade honestly when data is missing or contradictory? | UNKNOWN counts stay UNKNOWN (F); conflicting observations all listed (G) | Matcher rules; UNKNOWN-never-upgraded invariant; conflict reporting |
| **Audience invariance** | Does simplifying for a different reader change any conclusion? | Status / evidence-ref / count digests equal across 4 modes (Experiment A) | Presentation as a pure re-rendering; invariance digests |
| **Publication safety** | Is a flawed explanation withheld? | Guardrail PASS / FLAG / BLOCK agreement with verdict profile | `_guardrail` policy (block: ungrounded, any contradiction, rate ≥ 0.5; flag: any unknown) |

A sixth, *usability* dimension (comprehension measured by user study) is
deliberately **not claimed** — see D.8.

### C.2 Evaluation setup

- **Testbed**: fully synthetic banking sandbox; six scenarios chosen as
  evaluation *cases*, not random inputs — A clean path, B policy violation,
  C anomaly only, D violation + anomaly, F missing evidence, G conflicting
  evidence.
- **Two formal experiments** (final-phase experiment suite, executed in CI):
  *Experiment A — adaptive presentation consistency*; *Experiment B — SLM
  rationale honesty*.
- **Three test levels**: ~308 unit/e2e tests (no network, no model calls);
  integration tests that exercise the real local SLM path when the daemon is
  available and skip otherwise; GUI wiring tests that drive every event
  through Gradio's real postprocess layer. **334 tests** total across 26
  files.
- **Representative live evaluation**: each report's numbers were verified
  end-to-end through the HTTP GUI, not only in-process.

### C.3 Experiment A — cross-audience consistency

The same validated report is presented in all four audience modes
(TECHNICAL / BUSINESS / NORMAL / ELI5); mode count = 4 × runs. Effectiveness
criterion: **no conclusion may change with the audience**. The evaluation
compares status digests, evidence-reference digests and counts across modes
and requires exact equality, while prose length and vocabulary differ. The
GUI displays the invariance digest after every run, making the property
*visible to the evaluator*, not just asserted by the test-suite.

### C.4 Experiment B — honesty of the model-backed component

With no SLM provider configured, the rationale layer must (i) compose the
rationale deterministically from the evidence packet, (ii) set
`llm_used=False`, and (iii) report **zero tokens**. Effectiveness criterion:
*the honest default path is normal operation, not an error state*. The same
holds when the daemon is slow or unreachable (GUI timeout → fallback),
and when the model's citations do not resolve (quarantine → `grounded=False`
→ guardrail can never PASS).

### C.5 Evaluating honest failure (scenarios F and G)

Failure handling is treated as an explainability capability, not an edge
case:

- **F (missing evidence)**: no state was recorded → requirements, claims and
  counterfactuals resolve to UNKNOWN; the report says *what is missing and
  why*; nothing is upgraded to a verdict. Metric: count of UNKNOWN results =
  expected maxima, count of fabricated values = 0.
- **G (conflicting evidence)**: two opposite recorded results → status
  UNKNOWN with **all** observations listed as conflicting evidence. Metric:
  conflicts surfaced = conflicts present; conflicts silently resolved = 0.

### C.6 Quantitative results (representative run, scenario B)

| Indicator | Value |
|---|---|
| Requirements matched | 3 (1 VIOLATED, 2 SATISFIED, 0 UNKNOWN) |
| Claims checked / verdicts | 4 checked, all SUPPORTED (deterministic rules) |
| Unsupported-claim rate | 0.000 |
| Guardrail | PASS |
| Evidence index lines | 6 |
| Counterfactual | amount-reduction replay-confirmed (`replay performed: True`, uncertainty 0.05) |
| Latency | double-digit milliseconds end-to-end (MEASURED) |
| Model usage | 0 tokens (deterministic fallback; USD basis UNKNOWN — reported as unknown, not 0) |

Scenario-level qualitative results: A → PASS with all-SATISFIED; C/D →
anomaly summaries with causal relation stated as *"none (never causal)"*;
F → UNKNOWN end-to-end; G → UNKNOWN with conflicting evidence listed.

### C.7 Methodological insights

1. **The evaluation target shifts from model to process.** For agentic
   systems the auditable ground truth is the execution trace; effectiveness
   is then measurable *without* model internals, which makes the method
   applicable to any observable agent.
2. **Explanations need a guardrail of their own.** Verdict counts alone do
   not stop a bad explanation from being shown; a single, explicit,
   test-pinned publication policy (PASS/FLAG/BLOCK) converts validation
   into action.
3. **UNKNOWN is a result, not a failure.** Scoring frameworks that force a
   binary verdict inflate apparent accuracy; carrying UNKNOWN end-to-end and
   *measuring* it (scenario F) evaluates honesty directly.
4. **Invariance is a checkable property.** "The explanation is clear for
   everyone" is subjective; cross-mode digest equality turns audience
   adaptation into a falsifiable claim (Experiment A).
5. **Model contributions must be measured by subtraction.** Experiment B
   evaluates the SLM by what happens *without* it and by how much of its
   output survives grounding — the only robust way to credit an LLM's
   contribution to a deterministic pipeline.
6. **Replay-confirmation is a faithfulness probe for counterfactuals.**
   Labeling each counterfactual *replay-confirmed* vs *prediction only*
   (with uncertainty) gives an objective faithfulness marker that purely
   generative explainers lack.
7. **The test-suite is the methodology, executable.** Every honesty rule
   (no fabricated evidence, no UNKNOWN upgrades, no causal claims, basis
   labels) is encoded as a test, so the evaluation is reproducible and
   survives maintenance.
8. **Basis-labelled metrics prevent self-deception in evaluation.** Reporting
   cost as "0 tokens / USD unknown" instead of "0 USD", and latency as
   UNKNOWN when untimed, keeps the evaluation instrument from flattering the
   system under test.

### C.8 Threats to validity of this evaluation (stated openly)

1. Synthetic domain → results demonstrate mechanism behaviour, not
   performance on real banking traffic.
2. No human-subject study → comprehension/trust effects are argued from
   design, not measured.
3. One SLM (4B) → grounding/quarantine behaviour may differ for larger
   models.
4. Judge-based verification is optional and capped; deterministic rules
   dominate, so verdict quality is bounded by the rule set.
5. Scenarios are curated to cover known failure modes; unseen failure modes
   are not represented.

---

# Human-Centric & Ethical Perspectives
## (Promoting transparency, trust, and responsible AI)

---

## D. Human-Centric and Ethical Perspectives

### D.1 Transparency: making the agent's behaviour inspectable

- **Complete audit trail**: every action is recorded as an immutable,
  provenance-tagged event; the evidence index (E1…En) is shown to every
  audience and included in exports.
- **Epistemic transparency**: the audit renderings (`to_text()`, JSON
  export) label every line FACT / INFERENCE / UNKNOWN; the readable panels
  frame fact vs. guess by section (counterfactuals announced as
  *predictions*, analogies marked *"(not evidence)"*) — the reader is never
  left guessing which statements are observations.
- **Process transparency**: the 15-section report exposes what was checked,
  what was found, what was validated, what is uncertain and what the
  framework's limitations are — including its own performance profile.
- **Machine-readable accountability**: JSON/text exports preserve statuses,
  verdicts, guardrail decisions and basis labels for regulators and
  downstream audit systems.

### D.2 Trust: earned and bounded, not asserted

TRACE-XAI's position is *calibrated trust*, not maximal trust:

- **Trust is earned** through per-claim verification: each user-facing claim
  carries evidence references, a verdict, a verification method and a
  confidence value; the four-mode invariance digest is displayed so the
  reader can see that simplification did not change any conclusion.
- **Trust is bounded** by the guardrail: a contradictory or ungrounded
  explanation is BLOCKed, an uncertain one is FLAGged — the system will show
  a plainer, honest explanation rather than a fluent unreliable one.
- **No anthropomorphic over-reach**: the SLM is quarantined — it *words* the
  explanation, it never *decides* anything; verdicts are deterministic.
  Hidden chain-of-thought is never requested (`think: false`), so no
  fabricated "reasoning theatre" is shown to users.
- **Uncertainty is communicated, not hidden**: UNKNOWN statuses, conflict
  lists, counterfactual uncertainty values and metric basis labels are all
  visible in the UI — a user is never given false precision.

### D.3 Human oversight and control

- The framework is **decision-support**: it explains; it does not act on
  accounts and it does not recommend transactions.
- **Real human input is distinguishable from simulation**: CSAT/retention is
  `NOT_COLLECTED` until a person records a rating during a session;
  simulated scores are separate, explicitly labelled fields with their
  formula shown; the After-Effects example inputs are labelled
  *simulated* on screen.
- **Evaluator control**: the SLM toggle is opt-in with a visible timeout;
  turning it off is the fully honest deterministic path (Experiment B).

### D.4 Data ethics and safety

- **All synthetic, all local**: the banking data is generated on the
  machine; no customer data is processed or transmitted; the only inference
  runtime is a local 4B model on Ollama — a deliberately small model that
  keeps on-premise execution practical (data sovereignty + lower
  environmental cost).
- **External providers exist only as an explicitly configured option** and
  are covered by a written safety policy (no credentials in code, isolated
  configuration, integration tests opt-in).
- Unit tests never make model/network calls — the evaluation infrastructure
  itself cannot leak data or generate cost.

### D.5 Communication ethics

- **No causal over-claiming**: metric→trace links are labelled ASSOCIATION
  or HYPOTHESIS; satisfaction proxies (CSAT/retention) are *never* treated
  as evidence about factual claims; anomaly–violation relations are stated
  as *"none (never causal)"* when they merely co-occur.
- **No fabricated confidence**: unmeasured metrics are UNKNOWN, not 0; an
  untimed stage reports UNKNOWN latency; a fallback run reports 0 tokens —
  never an invented cost.
- **Plain language without distortion**: ELI5/NORMAL modes simplify wording
  while carrying the full evidence index and statuses; the presentation
  invariant forbids an UNKNOWN from being upgraded by simplification.
- **Limitations are part of the output**: every report embeds an explicit
  limitations section (e.g. "all data synthetic"), so the artefact cannot be
  read as more than it is.

### D.6 Responsible-AI principle → mechanism map

| Principle | Mechanism in TRACE-XAI | Visible evidence |
|---|---|---|
| Transparency | Evidence index, 15-section report, epistemic labels in audit layer | Tab 2/3/7 of the GUI; JSON/text export |
| Explainability | Q1–Q3 stages + validated rationale, 4 audience modes | Explanations tab |
| Accountability | Per-claim verdicts, guardrail decision log, provenance-tagged events | Claim checks + audit tab |
| Safety (of communication) | Guardrail BLOCK/FLAG; quarantine of ungrounded claims | Guardrail banner; `grounded` flag |
| Human oversight | Decision-support only; real vs simulated metrics labelled; opt-in SLM | Metrics tab; example-input labels |
| Privacy / data sovereignty | Fully synthetic local data; local 4B SLM; no egress by default | Safety policy; provider design |
| Honesty / non-deception | UNKNOWN-never-upgraded, no causal claims, basis labels, no hidden CoT | Enforced by the test-suite |
| Inclusiveness | Lay (ELI5) and general modes for non-expert stakeholders | Audience selector |

### D.7 What the framework explicitly does NOT claim

Stated for ethical completeness (and pinned by tests):

1. It does not guarantee that the *agent's decisions* are fair or correct —
   it audits the honesty and traceability of the process and its
   explanation, not the fairness of outcomes.
2. It does not claim causal understanding of metrics — only evidence-linked
   association.
3. It does not claim real customer-satisfaction measurement — that exists
   only when a person records a rating.
4. It does not present any model-generated statement as evidence — model
   output that cannot cite the records is quarantined, not shown as fact.
5. It does not certify regulatory compliance — it produces the audit
   artefacts that support a compliance process.

### D.8 Future human-centric work

A human-subject evaluation (comprehension and calibrated-trust study across
the four audience modes), accessibility improvements (screen-reader-friendly
renderings, multilingual lay modes), and a longitudinal field study of the
After-Effects monitor — each designed to convert the currently *designed*
human-centric properties into *measured* ones.

---

*Source pointers for the appendix: evaluation → `tests/unit`, `tests/e2e`,
`experiments/` (final-phase experiment suite), scenario configs in
`configs/`; ethics mechanisms → `src/trace_xai/presentation/presenter.py`
(invariants), `src/trace_xai/validate/validator.py` (guardrail),
`src/trace_xai/rationale/generator.py` (quarantine),
`src/trace_xai/linkage/linker.py` (association/hypothesis),
`src/trace_xai/metrics/collector.py` (basis labels).*
