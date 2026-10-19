# TRACE-XAI

**TRACE-XAI** is an evidence-grounded Explainable AI framework for AI-agent behavior,
researched on a **fully simulated banking agent with completely synthetic/fake financial data**.

## What it explains

| # | Question | Initial method |
|---|----------|----------------|
| Q1 | Why did the agent choose this action? | Requirement-to-action matching |
| Q2 | What went wrong vs. requirements/expected behavior? | Hybrid requirement checking + trajectory anomaly detection |
| Q3 | What could have happened with another action? | Alternative-action comparison + controlled replay |
| Q4 | Is the explanation supported by evidence? | Evidence verification, guardrails, LLM/SLM-as-judge |

## Core principles (non-negotiable)

1. **Synthetic data only.** No real banking credentials, accounts, transactions, or
   personal financial data — ever. See `docs/safety.md`.
2. **No hidden chain-of-thought.** TRACE-XAI produces an explicit, evidence-grounded
   *decision rationale* derived from observable inputs, actions, tool results,
   requirements, and outcomes.
3. **No component may invent evidence.** Every claim in an explanation must be labelled
   `FACT`, `EVIDENCE`, `INFERENCE`, or `UNKNOWN` and (where required) reference
   verifiable evidence records by ID.

## Status

Phase 8 — integrated pipeline: done — `TraceXAIPipeline` composes all
validated Q1–Q4 components behind a single call
(`analyze_execution(task_id)`) and returns a 15-section `FinalXAIReport`
with per-stage timings, embedded sub-reports, and computed limitations;
9 end-to-end scenarios (success, violation, anomaly, combined, counterfactual,
missing/conflicting evidence, unsupported vs correct explanations);
189 tests passing. All four research questions Q1–Q4 are implemented.

## Quickstart

```bash
cd /home/franz/Documents/TRACE-XAI
uv venv --system-site-packages --python /usr/bin/python3.11 .venv   # already done
.venv/bin/python -m pytest                                          # run the test suite
```

One call from task to final XAI report:

```python
from trace_xai import analyze_execution          # or TraceXAIPipeline for reuse
report = analyze_execution("Transfer ₹10,000 to beneficiary BOB-001.")
print(report.to_text())                          # 15-section final report
report.validation.guardrail_decision             # PASS / FLAG / BLOCK
report.requirement_satisfaction                  # {'REQ-AMT-001': 'SATISFIED', ...}
```

With the real local SLM for the rationale text (needs the Ollama service;
verdicts stay deterministic either way):

```python
from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.rationale.provider import OllamaRationaleProvider
pipeline = TraceXAIPipeline(agent=RuleBasedAgent(),
                            rationale_provider=OllamaRationaleProvider())
result, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
report = pipeline.analyze_execution(task_id=report.task_id)   # single-call API
```

## Layout

```
docs/        architecture, safety policy, phase plan
configs/     bank_dataset.yaml · requirements_bank.yaml (registry), scenarios, judge prompts
src/trace_xai/
  banking/       state.py · dataset.py · environment.py · results.py   (Phase 1 ✔)
  agent/         parser.py · rule_agent.py · langchain_agent.py · langchain_tools.py (Phase 1 ✔)
  core/          trace.py (schema) · recorder.py                       (Phase 2 ✔)
  requirements/  model.py (registry + constraint types)                (Phase 3 ✔)
  explain/       matcher.py · decision.py (Q1) · anomaly.py · deviation.py (Q2)
                 · counterfactual.py (Q3)                                 (Phase 3-5 ✔)
  rationale/     context.py · template.py · provider.py · generator.py   (Phase 6 ✔)
  validate/      extract.py · rules.py · judge.py · validator.py         (Phase 7 ✔)
  pipeline.py    TraceXAIPipeline -> FinalXAIReport                       (Phase 8 ✔)
data/        synthetic datasets, traces (sample_transfer_success.json), requirements
experiments/ phase1_demo.py · phase2_sample_trace.py · phase3_demo.py · phase4_demo.py · phase5_demo.py · phase6_demo.py · phase7_demo.py · phase8_demo.py
tests/       unit (deterministic) + e2e (scenarios A-I) + integration (LLM, auto-skip)
```
