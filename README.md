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

Phase 2 — structured execution traces: done (trace schema, recorder, agent
integration, committed sample trace; 73 tests passing). Explanations, anomaly
detection, counterfactuals, and judges are **not implemented yet**.

## Quickstart

```bash
cd /home/franz/Documents/TRACE-XAI
uv venv --system-site-packages --python /usr/bin/python3.11 .venv   # already done
.venv/bin/python -m pytest                                          # run the test suite
```

## Layout

```
docs/        architecture, safety policy, phase plan
configs/     bank_dataset.yaml (synthetic accounts), scenarios, judge prompts
src/trace_xai/
  banking/       state.py · dataset.py · environment.py · results.py   (Phase 1 ✔)
  core/          trace.py (schema) · recorder.py                       (Phase 2 ✔)
  agent/         parser.py · rule_agent.py · langchain_agent.py · langchain_tools.py (Phase 1 ✔)
  requirements/  requirement registry + matching                        (Phase 2)
  explain/       Q1 decision, Q2 deviation, Q3 counterfactual           (Phase 2-4)
  validate/      Q4 verifier, guardrails, LLM/SLM judge                 (Phase 4)
  llm/           LLM client (local Ollama default, fully optional)
data/        synthetic datasets, traces (sample_transfer_success.json), requirements
experiments/ phase1_demo.py · phase2_sample_trace.py and future paper artifacts
tests/       unit (deterministic) + integration (LLM, auto-skip)
```
