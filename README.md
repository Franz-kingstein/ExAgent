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

Phase 5 — counterfactual explainability (Q3): done — minimal alternative-action
generation, deterministic sandbox simulation against recorded pre-action state,
and controlled replay in an isolated environment clone (explicitly recorded,
never blended with the traced execution); requirement comparison +
improves/worsens impact labelling; 126 tests passing. The LLM judge and final
explanation validator (Q4) are **not implemented yet**.

## Quickstart

```bash
cd /home/franz/Documents/TRACE-XAI
uv venv --system-site-packages --python /usr/bin/python3.11 .venv   # already done
.venv/bin/python -m pytest                                          # run the test suite
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
  validate/      Q4 verifier, guardrails, LLM/SLM judge                 (Phase 6)
  llm/           LLM client (local Ollama default, fully optional)
data/        synthetic datasets, traces (sample_transfer_success.json), requirements
experiments/ phase1_demo.py · phase2_sample_trace.py · phase3_demo.py · phase4_demo.py · phase5_demo.py
tests/       unit (deterministic) + integration (LLM, auto-skip)
```
