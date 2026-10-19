"""TRACE-XAI evaluator GUI (Gradio) — all capabilities wired to real code.

    ./.venv/bin/python app.py            # launches on http://127.0.0.1:7860

Built for non-technical evaluators: every panel says in plain language what
it shows, and every metric carries its unit and its basis (measured /
estimated / simulated / unknown). Nothing here is a placeholder — each
control calls the actual TRACE-XAI components (pipeline, presentation,
metrics, stability, linkage, export).

Capabilities (12): scenario run · trace inspection · all four explanations ·
mode selection · evidence & verification status · latency + cost · claim
lists · CSAT controls · After-Effects monitoring · metric->trace linkage ·
consolidated report · export (text/JSON files).

All data is synthetic (see docs/safety.py): the scenario traces are
generated locally, CSAT ratings are only what you explicitly record in this
session, and After-Effects runs on simulated observation inputs.
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.linkage import MetricChange, link_metric
from trace_xai.metrics import CSATRecorder, ModelPricing
from trace_xai.presentation import PresentationMode, present_all
from trace_xai.rationale.provider import OllamaRationaleProvider
from trace_xai.stability import (
    InterventionEvent,
    InterventionType,
    PostActionStabilityMonitor,
)

MODES = [m.value for m in PresentationMode]
INTERVENTION_KINDS = [k.value for k in InterventionType]

PLAIN = {
    "guardrail": "Guardrail: the system's stoplight for whether an "
                 "explanation may be shown as-is (PASS / FLAG / BLOCK).",
    "unsupported": "Unsupported-claim rate: share of claims the records "
                   "could not back (contradicted or unverifiable).",
    "agent_ms": "Agent execution time: how long the banking agent itself "
                "took to act (milliseconds).",
    "xai_ms": "XAI analysis time: how long the explanation pipeline took "
              "after the action, on the recorded trace (milliseconds).",
    "cost": "Model cost: real token counts from the provider. A local "
            "model has no price tag, so money cost is shown as unknown "
            "unless you supply rates.",
    "csat": "CSAT: satisfaction score. No real user study exists — real "
            "ratings appear only when someone records one here; the "
            "simulated score is a labelled demonstration.",
    "pas": "After-Effects: 'after the action, within this window, did "
           "anyone intervene?' It never claims the action was correct.",
}


# --------------------------------------------------------- scenario builders
def _scenario_a(pipeline):
    result, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
    return report, pipeline._traces[report.task_id], result


def _scenario_b(pipeline):
    result, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
    return report, pipeline._traces[report.task_id], result


def _scenario_c(pipeline):
    env = BankEnvironment.from_dataset()
    r = ExecutionRecorder(task_id="TASK-SYN-C",
                          task_text="Transfer an invalid amount twice.",
                          agent_id="rule-based-agent", trace_id="TRC-SYN-C")
    r.record_task_received()
    for _ in range(2):
        r.record_tool_invocation(action="transfer", tool_name="transfer",
                                 tool_input={"amount_inr": -5000,
                                             "beneficiary_id": "BOB-001"},
                                 state_before=env.summary())
        r.record_tool_result(action="transfer", tool_name="transfer",
                             tool_output={"success": False,
                                          "reason_code": "INVALID_AMOUNT",
                                          "account_id": "ACC-SYN-001"},
                             success=False)
    r.record_outcome("INVALID_AMOUNT", success=False)
    trace = r.finish()
    return pipeline.analyze_execution(trace=trace), trace, None


def _scenario_d(pipeline):
    result, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-002.")
    return report, pipeline._traces[report.task_id], result


def _scenario_f(pipeline):
    r = ExecutionRecorder(
        task_id="TASK-SYN-F",
        task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b", trace_id="TRC-SYN-F")
    r.record_task_received()
    r.record_tool_invocation(action="transfer", tool_name="transfer",
                             tool_input={"amount_inr": 10_000,
                                         "beneficiary_id": "BOB-001"},
                             state_before=None)
    r.record_tool_result(action="transfer", tool_name="transfer",
                         tool_output={"success": False,
                                      "reason_code": "INSUFFICIENT_FUNDS"},
                         success=False)
    r.record_outcome("INSUFFICIENT_FUNDS", success=False)
    trace = r.finish()
    return pipeline.analyze_execution(trace=trace), trace, None


def _scenario_g(pipeline):
    env = BankEnvironment.from_dataset()
    r = ExecutionRecorder(
        task_id="TASK-SYN-G",
        task_text="Transfer ₹5,000 to beneficiary BOB-001.",
        agent_id="rule-based-agent", trace_id="TRC-SYN-G")
    r.record_task_received()
    r.record_tool_invocation(action="transfer", tool_name="transfer",
                             tool_input={"amount_inr": 5_000,
                                         "beneficiary_id": "BOB-001"},
                             state_before=env.summary())
    first = r.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": True, "reason_code": "OK",
                     "account_id": "ACC-SYN-001"}, success=True)
    r.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS",
                     "account_id": "ACC-SYN-001"},
        success=False, conflict_with_event_id=first.event_id)
    r.record_outcome("CONFLICT", success=None)
    trace = r.finish()
    return pipeline.analyze_execution(trace=trace), trace, None


SCENARIOS = {
    "A — successful transfer (₹10,000)": _scenario_a,
    "B — limit violation (₹75,000)": _scenario_b,
    "C — trajectory anomaly (invalid amount twice)": _scenario_c,
    "D — violation + anomaly (₹75,000, new beneficiary)": _scenario_d,
    "F — missing evidence (no state recorded)": _scenario_f,
    "G — conflicting evidence (two opposite results)": _scenario_g,
}

_EXPORT_DIR = Path(tempfile.gettempdir()) / "trace_xai_exports"

# One-click examples for the After-Effects panel (simulated observation
# data the evaluator can insert, edit and evaluate — clearly not system data)
EXAMPLE_INTERVENTION = json.dumps(
    [{"at_ms": 25000, "kind": "HUMAN_OVERRIDE",
      "description": "operator reversed it"}], indent=2)
EXAMPLE_GAP = json.dumps(
    [{"start_ms": 0, "end_ms": 10000, "reason": "collector offline"}],
    indent=2)


def _fresh_state() -> dict:
    return {"report": None, "trace": None, "result": None,
            "csat": CSATRecorder(), "pas_events": [], "pas_gaps": []}


# ------------------------------------------------------------- handlers
def _ollama_available() -> bool:
    """Is the local Ollama daemon reachable? (checked before consulting it)"""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def run_scenario(scenario_name: str, custom_task: str, state: dict,
                 mode_name: str = "normal", use_slm: bool = False):
    """Run the selected scenario (or a custom task) and cache the report.

    ``use_slm`` consults the local Ollama model for the *rationale prose
    only* (verdicts stay citation-grounded either way). When it is off — or
    the daemon is unreachable — the deterministic fallback is used and the
    metrics honestly report 0 tokens.
    """
    provider = None
    slm_note = "- **Rationale source:** deterministic fallback (no model " \
               "consulted) — token usage 0 by construction"
    if use_slm:
        if _ollama_available():
            # shorter timeout than the library default: the GUI must not
            # freeze for minutes on a cold model load — a timeout falls back
            # to the deterministic path and says so
            provider = OllamaRationaleProvider(timeout=90.0)
            slm_note = ("- **Rationale source:** local SLM "
                        f"(`{provider.name}`) — token usage below is "
                        "provider-reported")
        else:
            slm_note = ("- **Rationale source:** deterministic fallback — "
                        "local Ollama was not reachable, so no model was "
                        "consulted (not an error)")
    pipeline = TraceXAIPipeline(
        agent=RuleBasedAgent(BankEnvironment.from_dataset()),
        rationale_provider=provider)
    state = dict(_fresh_state())
    if custom_task.strip():
        result, report = pipeline.run(custom_task.strip())
        trace = pipeline._traces[report.task_id]
    else:
        report, trace, result = SCENARIOS[scenario_name](pipeline)
    state.update(report=report, trace=trace, result=result)

    v = report.validation
    summary = [
        f"### Ran: {report.task}",
        f"- **Task id:** {report.task_id} · **Trace id:** {report.trace_id}",
        f"- **Agent action:** {' · '.join(report.agent_action[:2])}",
        f"- **Guardrail:** {v.guardrail_decision.value} — {PLAIN['guardrail']}",
        f"- **Claims supported / contradicted / unknown:** "
        f"{v.supported_claim_count} / {v.contradicted_claim_count} / "
        f"{v.unknown_claim_count}",
        f"- **Requirements:** "
        + (", ".join(f"{k}={s}" for k, s
                     in report.requirement_satisfaction.items())),
        slm_note,
    ]
    if custom_task.strip():
        summary.append("- *custom task; all data remains synthetic "
                       "(local sandbox, no external calls)*")
    # fill the downstream panels from the new state directly, so every tab
    # has real content the moment the run completes (no empty outputs)
    expl_text, expl_digest = explanation_view(mode_name, state)
    pas_msg = ("Run complete. Enter simulated observation data "
               "(interventions / gaps) below and press "
               "'Evaluate After-Effects'.")
    return ("\n".join(summary), _trace_rows(trace), state,
            expl_text, expl_digest, pas_msg)


def _trace_rows(trace) -> list[list]:
    rows = []
    for e in trace.events:
        rows.append([
            e.sequence, e.event_id,
            e.timestamp.isoformat() if e.timestamp else "(not tracked)",
            e.event_type.value, e.action or "",
            json.dumps(e.tool_input, default=str)[:80] if e.tool_input else "",
            json.dumps(e.tool_output, default=str)[:80] if e.tool_output else "",
            ("conflict with " + e.conflict_with_event_id)
            if e.conflict_with_event_id else "",
        ])
    return rows


def trace_view(state: dict) -> str:
    """Plain-language trace header (the event table comes from the run)."""
    trace = state.get("trace")
    if trace is None:
        return "Run a scenario first."
    lines = [f"Trace **{trace.trace_id}** (task {trace.task_id}) — "
             f"{len(trace.events)} recorded events, agent "
             f"`{trace.agent_id}`.", "",
             "Every value below was observed in the local synthetic "
             "sandbox; nothing is inferred at this layer."]
    return "\n".join(lines)


def explanation_view(mode: str, state: dict):
    """The selected audience mode + the cross-mode invariance digest."""
    report = state.get("report")
    if report is None:
        return "Run a scenario first.", ""
    presented = present_all(report)
    chosen = presented[PresentationMode(mode)]
    all_modes = present_report_digests(presented)
    header = (f"**Cross-mode check (same validated report):** {all_modes}\n\n"
              f"Evidence, claim texts, statuses and counts are copied from "
              f"the validated report — only the wording changes.\n\n---\n")
    return header + chosen.to_markdown(), all_modes


def present_report_digests(presented: dict) -> str:
    status_ok = len({str(e.status_digest()) for e in presented.values()}) == 1
    refs_ok = len({e.ref_digest() for e in presented.values()}) == 1
    counts_ok = len({str(e.counts) for e in presented.values()}) == 1
    return (f"statuses identical across all 4 modes: "
            f"{'✅' if status_ok else '❌'} · evidence refs identical: "
            f"{'✅' if refs_ok else '❌'} · counts identical: "
            f"{'✅' if counts_ok else '❌'}")


def evidence_view(state: dict):
    """Claims verdicts + the evidence index (the actual recorded values).

    Evidence is derived 100% from the local synthetic trace — no online
    lookups exist anywhere in this pipeline by design.
    """
    report = state.get("report")
    if report is None:
        return "Run a scenario first.", [], []
    v = report.validation
    header = (f"Evidence index ({len(report.evidence)} observed values, "
              f"listed below — all read from the local synthetic trace, "
              f"never from an online source). Each verified claim cites the "
              f"exact values it rests on; verdicts are copied from the "
              f"validation stage, never recomputed here.")
    index_rows = []
    for line in report.evidence:
        ref, _, detail = line.partition(" = ")
        index_rows.append([ref, detail])
    rows = [[c.claim, c.verification_status.value,
             c.verification_method.value, f"{c.confidence:.2f}",
             " · ".join(c.evidence) or "(none cited)",
             c.explanation] for c in v.verifications]
    return header, index_rows, rows


def metrics_view(state: dict, use_pricing: bool, prompt_price: float,
                 completion_price: float, simulate: bool):
    """Latency + cost + claim quality + CSAT, each with unit and basis."""
    report = state.get("report")
    if report is None:
        return "Run a scenario first."
    pricing = None
    if use_pricing:
        pricing = ModelPricing(
            usd_per_prompt_token=prompt_price,
            usd_per_completion_token=completion_price,
            source="entered in the GUI by the evaluator (estimate only)")
    m = report.metrics(pricing=pricing, simulate_csat=simulate,
                       csat=state["csat"])
    lat, cost, cq, csat = m.latency, m.cost, m.claim_quality, m.csat

    def basis(b): return f"`{b.value}`"
    lines = [
        "### Latency", f"- {PLAIN['agent_ms']} **{lat.agent_execution_ms}** "
        f"ms — basis {basis(lat.agent_basis)}",
        f"- {PLAIN['xai_ms']} **{lat.xai_analysis_ms}** ms "
        f"(stages: " + ", ".join(f"{k}={v}" for k, v in
                                 lat.xai_stage_ms.items()) + ")",
        f"- End-to-end: **{lat.end_to_end_ms}** ms — basis "
        f"{basis(lat.end_to_end_basis)}",
        "", "### Cost", f"- {PLAIN['cost']}",
        f"- Tokens (prompt / completion): **{cost.total_prompt_tokens}** / "
        f"**{cost.total_completion_tokens}** — basis {basis(cost.basis)}",
        (f"- Estimated money cost: **{cost.estimated_cost_usd}** USD"
         if cost.estimated_cost_usd is not None else
         "- Estimated money cost: **unknown** — real tokens were used but "
         "this model has no per-token price; supply pricing above to "
         "estimate"),
        ("- Pricing supplied: " + cost.pricing_source) if cost.pricing_known
        else "- Pricing not supplied → cost reported as unknown, not guessed",
    ]
    if cost.notes:
        lines.extend(f"  - note: {n}" for n in cost.notes)
    lines += [
        "", "### Explanation quality (claims)",
        f"- {PLAIN['unsupported']} **{cq.unsupported_claim_rate}** "
        f"(formula: {cq.formula})",
        f"- Contradicted (records say otherwise): **{cq.contradicted_count}**"
        f" · rate {cq.contradiction_rate}",
        f"- Unverifiable (records too thin — not a lie): "
        f"**{cq.unknown_count}** · rate {cq.unverifiable_rate}",
        "", "### Satisfaction (CSAT)",
        f"- {PLAIN['csat']}",
        f"- Real ratings recorded: **{csat.responses}** "
        f"(status {csat.status}, mean {csat.mean_rating})",
    ]
    if csat.simulated:
        s = csat.simulated
        lines.append(f"- **SIMULATED** mean: **{s.simulated_mean_rating}** "
                     f"/ 5 over {s.simulated_responses} synthetic responses "
                     f"(label: {s.label}; formula: `{s.formula}`)")
    if csat.simulated_retention:
        r = csat.simulated_retention
        lines.append(f"- **SIMULATED** retention: "
                     f"**{r.estimated_retention_rate}** (label: {r.label}; "
                     f"formula: `{r.formula}`)")
    return "\n".join(lines)


def csat_record(rating: int, state: dict):
    """Record one real satisfaction rating (1-5) for this session."""
    if state.get("report") is None:
        return "Run a scenario first.", state
    state["csat"].record(int(rating))
    n = state["csat"].response_count
    return (f"Recorded rating **{rating}**. This session now has **{n}** "
            f"real rating(s). Only explicitly recorded ratings ever appear "
            f"as real CSAT — nothing is fabricated.", state)


def linkage_view(metric: str, before: float | None, after: float | None,
                 req_scope: str, state: dict):
    report = state.get("report")
    if report is None:
        return "Run a scenario first."
    change = MetricChange(
        metric=metric or "custom_metric",
        before=before, after=after, unit="",
        basis="measured",
        related_requirement_ids=[r.strip() for r in req_scope.split(",")
                                 if r.strip()])
    return link_metric(report, change).to_text()


def pas_view(window_ms: float, observe_ms: float, event_json: str,
             gap_json: str, state: dict):
    """Evaluate After-Effects with the user's simulated observation inputs.

    Format for events: [{"at_ms": 25000, "kind": "HUMAN_OVERRIDE",
    "description": "..."}]  — gaps: [{"start_ms": 0, "end_ms": 10000,
    "reason": "..."}]. Timestamps are offsets from action completion.
    """
    trace = state.get("trace")
    if trace is None:
        return "Run a scenario first."
    anchor = None
    for e in reversed(trace.events):
        if e.timestamp is not None:
            anchor = e.timestamp
            break
    if anchor is None:
        return ("This trace has no timestamps, so After-Effects cannot be "
                "anchored (reported as a limitation, not guessed).")

    def parse(raw: str, kind: str) -> list[dict]:
        raw = raw.strip()
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{kind} is not valid JSON: {exc}") from exc
        if not isinstance(value, list):
            raise ValueError(f"{kind} must be a JSON list")
        return value

    try:
        events = parse(event_json, "interventions")
        gaps = parse(gap_json, "gaps")
    except ValueError as exc:
        return f"Input error: {exc}"

    monitor = PostActionStabilityMonitor(anchor, window_ms=float(window_ms))
    for item in events:
        try:
            monitor.observe(InterventionEvent(
                timestamp=anchor + timedelta(milliseconds=float(item["at_ms"])),
                kind=InterventionType(item.get("kind", "OTHER")),
                description=item.get("description", ""),
                source="simulated (entered in the GUI)"))
        except (KeyError, ValueError) as exc:
            return f"Intervention entry error: {exc!r} — expected " \
                   f"at_ms/kind/description with kind in {INTERVENTION_KINDS}"
    for item in gaps:
        try:
            monitor.record_gap(
                anchor + timedelta(milliseconds=float(item["start_ms"])),
                anchor + timedelta(milliseconds=float(item["end_ms"])),
                reason=item.get("reason", "unspecified"))
        except (KeyError, ValueError) as exc:
            return f"Gap entry error: {exc!r} — expected " \
                   f"start_ms/end_ms/reason"
    result = monitor.close(observe_until=anchor
                           + timedelta(milliseconds=float(observe_ms)))
    return ("### After-Effects result (simulated observation)\n\n"
            + result.to_text()
            + "\n\n*Entered observation data is simulated; the verdict is "
              "bounded by the window you set.*")


def consolidated_view(state: dict):
    report = state.get("report")
    if report is None:
        return "Run a scenario first."
    parts = [report.to_text(), ""]
    for mode in MODES:
        parts.append(present_all(report)[PresentationMode(mode)].to_text())
        parts.append("")
    m = report.metrics(simulate_csat=True)
    parts.append("=== METRICS (simulated CSAT attached, labelled) ===")
    parts.append(m.to_text() if hasattr(m, "to_text") else m.model_dump_json(
        indent=2))
    return "\n".join(parts)


def export_files(state: dict):
    """Write report / presentation / metrics files and offer them."""
    report = state.get("report")
    if report is None:
        return None, None, "Run a scenario first."
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{report.task_id}_{report.trace_id}"
    txt_path = _EXPORT_DIR / f"{stem}_report.txt"
    txt_path.write_text(consolidated_view(state))
    json_path = _EXPORT_DIR / f"{stem}_report.json"
    payload = report.model_dump(mode="json")
    payload["presentations"] = {m: present_all(report)[
        PresentationMode(m)].model_dump(mode="json") for m in MODES}
    payload["metrics"] = report.metrics(simulate_csat=True).model_dump(
        mode="json")
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    return str(txt_path), str(json_path), (
        f"Exported two files for {report.task_id} (text + JSON with all "
        f"four presentations and metrics).")


# ------------------------------------------------------------------ layout
def build_app() -> gr.Blocks:
    with gr.Blocks(title="TRACE-XAI — explainable banking agent") as demo:
        gr.Markdown(
            "# TRACE-XAI — evaluate an explainable banking agent\n"
            "Synthetic data only. Every number shows its unit and where it "
            "came from (measured / estimated / simulated / unknown).")
        state = gr.State(_fresh_state())

        with gr.Tab("1 · Run a scenario"):
            scenario = gr.Dropdown(choices=list(SCENARIOS), value=
                                   list(SCENARIOS)[0],
                                   label="Pick a synthetic scenario")
            custom = gr.Textbox(
                label="Or type your own task (synthetic sandbox)",
                placeholder="Transfer ₹20,000 to beneficiary BOB-001.")
            run_btn = gr.Button("Run", variant="primary")
            use_slm = gr.Checkbox(
                label="Consult local SLM (Ollama qwen3.5:4b) for the "
                      "rationale prose — records real provider token usage "
                      "(verdicts stay citation-grounded either way)",
                value=False)
            run_out = gr.Markdown()
            trace_head = gr.Markdown()
            trace_rows = gr.Dataframe(
                label="Recorded trace events (step by step)",
                headers=["step", "event id", "timestamp", "type", "action",
                         "tool input", "tool output", "notes"],
                interactive=False)

        with gr.Tab("2 · Explanations (4 audience modes)"):
            gr.Markdown(
                "One validated analysis, four wordings. Evidence, claim "
                "texts, statuses and counts never change between modes — "
                "only the language does. UNKNOWN never becomes 'supported' "
                "by simplifying.")
            mode = gr.Radio(choices=MODES, value="normal",
                            label="Explanation mode")
            expl_digest = gr.Markdown(
                value="Run a scenario to see the cross-mode invariance check.")
            expl_out = gr.Markdown()

        with gr.Tab("3 · Evidence & claims"):
            ev_head = gr.Markdown()
            ev_index = gr.Dataframe(
                label="Observed evidence values (from the local synthetic "
                      "trace)",
                headers=["ref", "recorded value"], interactive=False)
            claims = gr.Dataframe(
                label="Verified claims (verdicts from the validation stage)",
                headers=["claim", "status", "method", "confidence",
                         "evidence cited", "why"],
                interactive=False)

        with gr.Tab("4 · Metrics"):
            gr.Markdown(
                "Every metric below states its unit and basis. "
                "**measured** = observed in this run · **estimated** = "
                "measured inputs + a documented formula · **simulated** = "
                "labelled stand-in · **unknown** = not determinable, "
                "reported as unknown.")
            with gr.Row():
                use_pricing = gr.Checkbox(
                    label="Supply pricing to estimate money cost", value=False)
                prompt_price = gr.Number(label="USD per prompt token",
                                         value=0.0, minimum=0)
                completion_price = gr.Number(label="USD per completion token",
                                             value=0.0, minimum=0)
            simulate = gr.Checkbox(
                label="Attach SIMULATED CSAT/retention (clearly labelled)",
                value=False)
            metrics_out = gr.Markdown()
            with gr.Accordion(
                    "CSAT controls (real ratings only what you enter)",
                    open=False):
                rating = gr.Slider(minimum=1, maximum=5, step=1, value=4,
                                   label="Your rating (1 = very dissatisfied, "
                                         "5 = very satisfied)")
                rate_btn = gr.Button("Record this rating")
                csat_out = gr.Markdown()

        with gr.Tab("5 · After-Effects (post-action stability)"):
            gr.Markdown(PLAIN["pas"] + " Enter simulated observation data "
                                       "below (offsets in ms from action "
                                       "completion).")
            with gr.Row():
                window_ms = gr.Number(label="Monitoring window (ms)",
                                      value=60000, minimum=1)
                observe_ms = gr.Number(label="Observe until (ms after action)",
                                       value=60000, minimum=0)
            events_json = gr.Textbox(
                label='Interventions (JSON list, e.g. [{"at_ms": 25000, '
                      '"kind": "HUMAN_OVERRIDE", "description": "operator '
                      'reversed it"}])',
                value="[]", lines=3)
            gaps_json = gr.Textbox(
                label='Observation gaps (JSON list, e.g. [{"start_ms": 0, '
                      '"end_ms": 10000, "reason": "collector offline"}])',
                value="[]", lines=2)
            with gr.Row():
                ex_intervene = gr.Button("Insert example intervention")
                ex_gap = gr.Button("Insert example gap")
                ex_clear = gr.Button("Clear both")
            pas_btn = gr.Button("Evaluate After-Effects")
            pas_out = gr.Markdown()

        with gr.Tab("6 · Metric → trace linkage"):
            gr.Markdown(
                "Shows which recorded events, claims and evidence a metric "
                "movement can legitimately reach. The chain is labelled "
                "**association** or **hypothesis** — never causation.")
            with gr.Row():
                link_metric_in = gr.Textbox(label="Metric name",
                                            value="unsupported_claim_rate")
                link_before = gr.Number(label="Before (optional)",
                                        value=None)
                link_after = gr.Number(label="After (optional)", value=None)
                link_scope = gr.Textbox(
                    label="Requirement ids (comma-separated, optional)",
                    value="REQ-AMT-001")
            link_btn = gr.Button("Build chain")
            link_out = gr.Markdown()

        with gr.Tab("7 · Consolidated report & export"):
            report_out = gr.Textbox(label="Full report (all sections, all "
                                          "modes, metrics)",
                                    lines=18, interactive=False)
            with gr.Row():
                export_btn = gr.Button("Export files (text + JSON)",
                                       variant="primary")
            export_status = gr.Markdown()
            export_txt = gr.File(label="Text report")
            export_json = gr.File(label="JSON (report + 4 presentations + metrics)")

        run_btn.click(run_scenario,
                      inputs=[scenario, custom, state, mode, use_slm],
                      outputs=[run_out, trace_rows, state, expl_out,
                               expl_digest, pas_out])
        run_btn.click(trace_view, inputs=[state], outputs=[trace_head])
        # populate the other tabs from the fresh state after each run
        run_btn.click(evidence_view, inputs=[state],
                      outputs=[ev_head, ev_index, claims])
        run_btn.click(metrics_view,
                      inputs=[state, use_pricing, prompt_price,
                              completion_price, simulate],
                      outputs=[metrics_out])
        run_btn.click(consolidated_view, inputs=[state],
                      outputs=[report_out])
        mode.change(explanation_view, inputs=[mode, state],
                    outputs=[expl_out, expl_digest])
        rate_btn.click(csat_record, inputs=[rating, state],
                       outputs=[csat_out, state])
        for component in (use_pricing, prompt_price, completion_price,
                          simulate):
            component.change(metrics_view,
                             inputs=[state, use_pricing, prompt_price,
                                     completion_price, simulate],
                             outputs=[metrics_out])
        pas_btn.click(pas_view,
                      inputs=[window_ms, observe_ms, events_json, gaps_json,
                              state],
                      outputs=[pas_out])
        ex_intervene.click(lambda: EXAMPLE_INTERVENTION,
                           outputs=[events_json])
        ex_gap.click(lambda: EXAMPLE_GAP, outputs=[gaps_json])
        ex_clear.click(lambda: ("[]", "[]"),
                       outputs=[events_json, gaps_json])
        link_btn.click(linkage_view,
                       inputs=[link_metric_in, link_before, link_after,
                               link_scope, state],
                       outputs=[link_out])
        export_btn.click(export_files, inputs=[state],
                         outputs=[export_txt, export_json, export_status])
        # first paint: default scenario summary so tabs are never empty
        demo.load(lambda: ("Run a scenario to begin.", [], _fresh_state(),
                           "Run a scenario first.", "Run a scenario first."),
                  outputs=[run_out, trace_rows, state, expl_out, pas_out])
        demo.load(metrics_view,
                  inputs=[state, use_pricing, prompt_price, completion_price,
                          simulate],
                  outputs=[metrics_out])
        demo.load(evidence_view, inputs=[state],
                  outputs=[ev_head, ev_index, claims])
        demo.load(consolidated_view, inputs=[state], outputs=[report_out])
    return demo


def main() -> None:
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
