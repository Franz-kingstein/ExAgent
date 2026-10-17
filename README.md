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

Phase 6 — SLM rationale layer: done — structured prompt over a pre-computed
evidence packet (no chain-of-thought requested or exposed), replaceable model
provider (local Ollama default, external endpoint isolated, deterministic
fallback), five-section user-facing rationale with citation grounding that
quarantines any invented evidence id; 151 tests passing (4 real-model
integration tests). The final explanation judge/validator (Q4) is **not
implemented yet**.

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
  rationale/     context.py · template.py · provider.py · generator.py   (Phase 6 ✔)
  validate/      Q4 verifier, guardrails, LLM/SLM judge                 (Phase 7)
data/        synthetic datasets, traces (sample_transfer_success.json), requirements
experiments/ phase1_demo.py · phase2_sample_trace.py · phase3_demo.py · phase4_demo.py · phase5_demo.py · phase6_demo.py
tests/       unit (deterministic) + integration (LLM, auto-skip)
```
