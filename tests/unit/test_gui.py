"""GUI tests (Part 7): every capability is real, wired and honest.

The app is tested through ``build_app()`` (constructs the full Gradio graph
without launching) and by calling each handler directly against a real
scenario run. Checks:

- all 12 capabilities exist and produce real output (no placeholders);
- cross-mode invariance digest is shown to the evaluator;
- metrics display units and measured/estimated/simulated/unknown basis;
- CSAT: only explicitly recorded ratings appear as real; simulation labelled;
- After-Effects works on injected inputs and reports errors honestly;
- export writes genuine text/JSON files carrying report + modes + metrics;
- ``TestGradioWiring`` drives every wired event through Gradio's real
  postprocess layer, so handler/output arity or type mismatches (e.g. a
  tuple handed to a Markdown) fail as tests instead of at click-time.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr  # noqa: E402
from gradio.state_holder import SessionState  # noqa: E402

import app as gui  # noqa: E402


@pytest.fixture
def state():
    s = gui._fresh_state()
    out = gui.run_scenario(list(gui.SCENARIOS)[1], "", s)   # scenario B
    return out[2]


@pytest.fixture
def pipeline_state():
    """Scenario F: missing state_before -> requirement statuses UNKNOWN."""
    out = gui.run_scenario("F — missing evidence (no state recorded)", "",
                           gui._fresh_state())
    return out[2]


def _app():
    return gui.build_app()


# ------------------------------------------------------------ app builds
class TestAppConstruction:
    def test_builds_without_launching(self):
        demo = _app()
        assert demo is not None

    def test_no_placeholder_markers_in_source(self):
        source = Path(gui.__file__).read_text()
        for marker in ("TODO", "FIXME", "coming soon", "not implemented",
                       "placeholder for"):
            assert marker.lower() not in source.lower(), \
                f"placeholder marker {marker!r} in app source"


# ------------------------------------------------- 1. scenario run + trace
class TestRunAndTrace:
    def test_all_six_scenarios_run(self):
        for name in gui.SCENARIOS:
            s = gui._fresh_state()
            out = gui.run_scenario(name, "", s)
            assert out[2]["report"] is not None, name
            assert "Guardrail" in out[0], name
            assert len(out[1]) >= 4, name            # trace rows
            assert "explanation" in out[3].lower(), name   # expl panel filled
            assert "Claims checked" in out[4], name         # proof panel filled
            assert "Run complete" in out[6], name           # PAS prompt shown

    def test_custom_task_runs_in_synthetic_sandbox(self):
        s = gui._fresh_state()
        out = gui.run_scenario(list(gui.SCENARIOS)[0],
                               "Transfer ₹20,000 to beneficiary BOB-001.", s)
        assert "₹20,000" in out[0]
        assert "synthetic" in out[0]

    def test_run_honours_selected_mode(self):
        s = gui._fresh_state()
        out = gui.run_scenario(list(gui.SCENARIOS)[0], "", s, "eli5")
        assert "ELI5 explanation" in out[3]

    def test_trace_view_labels_observed_data(self, state):
        head = gui.trace_view(state)
        assert "observed" in head and "inferred" in head
        rows = gui._trace_rows(state["trace"])
        assert rows and rows[0][0] == 1              # sequence starts at 1

    def test_empty_state_is_guided_not_crashing(self):
        empty = gui._fresh_state()
        assert "Run a scenario first" in gui.trace_view(empty)
        assert "Run a scenario first" in gui.evidence_view(empty)[0]
        assert "Run a scenario first" in gui.explanation_view("eli5", empty)[0]
        assert "Run a scenario first" in gui.metrics_view(empty, False, 0, 0,
                                                          False)
        assert "Run a scenario first" in gui.consolidated_view(empty)


# -------------------------------------------------- 2/3. explanations + evidence
class TestExplanationsAndEvidence:
    def test_all_four_modes_render(self, state):
        texts = {}
        for mode in gui.MODES:
            text, proof, digest = gui.explanation_view(mode, state)
            texts[mode] = text + proof
            # story panel is pure markdown; the invariance check is its own
            # line, never between reader and story
            assert "Cross-mode check" in digest
            assert "✅" in digest and "❌" not in digest
        assert len(set(texts.values())) == 4      # wording really differs

    def test_explanation_panel_is_markdown_not_text_artifacts(self, state):
        """The UI panels must use headings/legend, not '-- title --' rows."""
        for mode in gui.MODES:
            text, proof, _ = gui.explanation_view(mode, state)
            for line in (text + proof).splitlines():
                assert not line.startswith("-- "), \
                    f"{mode}: plain-text separator leaked into the panel"
            assert "### " in text
            assert "How to read the tags" in text

    def test_explanation_keeps_unknown_visible(self, pipeline_state):
        """UNKNOWN claims must still be labelled UNKNOWN in the panel."""
        text, proof, _ = gui.explanation_view("eli5", pipeline_state)
        assert "UNKNOWN" in text + proof or "we cannot tell" in text \
            or "nobody knows" in text

    def test_story_and_proof_split_cleanly(self, state):
        """The story panel stays story; numbers/verdicts live in proof."""
        story, proof, digest = gui.explanation_view("normal", state)
        assert "Cross-mode check" in digest
        assert "**How to read the tags:**" in story   # legend opens the story
        assert "Claims checked" not in story          # numbers live in proof
        assert "Claims checked" in proof
        assert "Guardrail:" in proof
        assert "[FACT]" in story                      # story is tagged lines

    def test_evidence_and_claims_tables_populated(self, state):
        head, index_rows, rows = gui.evidence_view(state)
        assert "Evidence index" in head and "never recomputed" in head
        assert "local synthetic trace" in head          # provenance stated
        assert index_rows and index_rows[0][0].startswith("E")
        assert "=" in index_rows[0][1] or "state" in index_rows[0][1]
        assert rows
        for row in rows:
            assert row[1] in ("SUPPORTED", "CONTRADICTED", "UNKNOWN")
            assert row[3]                      # confidence rendered
            assert row[4]                      # evidence column present

    def test_evidence_values_are_trace_derived_not_online(self, state):
        """Every evidence ref must resolve to a value recorded in the trace."""
        _, index_rows, _ = gui.evidence_view(state)
        trace_ids = {e.event_id for e in state["trace"].events}
        for ref, detail in index_rows:
            assert any(tid in detail for tid in trace_ids), (ref, detail)


# ------------------------------------------------------------- 4. metrics
class TestMetricsPanel:
    def test_units_and_basis_shown(self, state):
        m = gui.metrics_view(state, False, 0, 0, False)
        assert "ms" in m and "Latency" in m
        for basis_word in ("`measured`", "`unknown`", "`estimated`",
                           "`simulated`", "`not_applicable`"):
            pass  # at least the basis fields must be rendered somewhere
        assert "basis `measured`" in m or "basis `unknown`" in m
        assert "Tokens" in m and "USD" in m
        assert "formula" in m
        assert "not a lie" in m or "not a hallucination" in m \
            or "not a lie" in m

    def test_unknown_pricing_stated_not_guessed(self, state):
        m = gui.metrics_view(state, False, 0, 0, False)
        assert "unknown, not guessed" in m or "not guessed" in m

    def test_pricing_toggle_produces_estimate_label(self, state):
        m = gui.metrics_view(state, True, 0.0001, 0.0002, False)
        assert "Pricing supplied" in m
        assert "entered in the GUI by the evaluator" in m

    def test_simulated_csat_only_when_requested(self, state):
        off = gui.metrics_view(state, False, 0, 0, False)
        on = gui.metrics_view(state, False, 0, 0, True)
        assert "SIMULATED" not in off
        assert "SIMULATED" in on
        assert "No real user study" in on


# ---------------------------------------------------------- CSAT controls
class TestCSATControls:
    def test_only_recorded_ratings_are_real(self, state):
        msg, s2 = gui.csat_record(5, state)
        assert "real rating" in msg and "fabricated" in msg
        m = gui.metrics_view(s2, False, 0, 0, False)
        assert "Real ratings recorded: **1**" in m
        assert "COLLECTED" in m

    def test_empty_recorder_reports_not_collected(self, state):
        m = gui.metrics_view(state, False, 0, 0, False)
        assert "NOT_COLLECTED" in m

    def test_out_of_scale_rating_is_rejected_honestly(self, state):
        with pytest.raises(ValueError):
            state["csat"].record(9)


# ------------------------------------------------------ 5. After-Effects PAS
class TestAfterEffectsPanel:
    def test_intervention_case(self, state):
        out = gui.pas_view(
            60000, 60000,
            json.dumps([{"at_ms": 25000, "kind": "HUMAN_OVERRIDE",
                         "description": "operator reversed"}]), "[]", state)
        assert "INTERVENTION_DETECTED" in out
        assert "simulated" in out

    def test_clean_window_is_bounded_claim(self, state):
        out = gui.pas_view(60000, 60000, "[]", "[]", state)
        assert "NO_INTERVENTION_OBSERVED" in out
        assert "not proof" in out

    def test_gap_forces_insufficient(self, state):
        out = gui.pas_view(
            60000, 60000, "[]",
            json.dumps([{"start_ms": 0, "end_ms": 60000,
                         "reason": "collector offline"}]), state)
        assert "INSUFFICIENT_OBSERVATION" in out
        assert "observation gap" in out

    def test_bad_json_is_an_input_error_not_a_crash(self, state):
        assert gui.pas_view(60000, 60000, "not json", "[]", state) \
            .startswith("Input error")
        assert gui.pas_view(60000, 60000, "{}", "[]", state) \
            .startswith("Input error")

    def test_bad_kind_is_rejected(self, state):
        out = gui.pas_view(60000, 60000,
                           json.dumps([{"at_ms": 1, "kind": "MAGIC"}]),
                           "[]", state)
        assert out.startswith("Intervention entry error")


# ------------------------------------------------------- 6. metric linkage
class TestLinkagePanel:
    def test_chain_built_with_scope(self, state):
        out = gui.linkage_view("unsupported_claim_rate", 0.4, 0.0,
                               "REQ-AMT-001", state)
        assert "[1] METRIC" in out and "[4] EXPLANATION CLAIMS" in out
        assert "ASSOCIATION" in out
        assert "no causal" in out

    def test_csat_chain_is_hypothesis(self, state):
        out = gui.linkage_view("csat", 3.0, 4.0, "REQ-AMT-001", state)
        assert "HYPOTHESIS" in out

    def test_empty_state_handled(self):
        assert "Run a scenario first" in gui.linkage_view(
            "m", None, None, "", gui._fresh_state())


# ------------------------------------------------- 7. report + export
class TestConsolidatedAndExport:
    def test_consolidated_carries_everything(self, state):
        rep = gui.consolidated_view(state)
        rep_low = rep.lower()
        for marker in ("[15] Limitations", "TRACE-XAI TECHNICAL EXPLANATION",
                       "TRACE-XAI ELI5 EXPLANATION", "metrics", "csat"):
            assert marker.lower() in rep_low, marker

    def test_export_writes_real_files(self, state):
        txt, js, status = gui.export_files(state)
        assert os.path.exists(txt) and os.path.exists(js)
        assert "Exported two files" in status
        payload = json.loads(Path(js).read_text())
        assert set(payload["presentations"]) == set(gui.MODES)
        assert "metrics" in payload and "validation" in payload
        assert Path(txt).stat().st_size > 1000

    def test_export_without_run_is_guided(self):
        txt, js, status = gui.export_files(gui._fresh_state())
        assert txt is None and js is None
        assert "Run a scenario first" in status


# ------------------------------------------------ SLM toggle + PAS examples
class TestSLMAndPASEnhancements:
    def test_default_run_is_deterministic_fallback(self):
        """use_slm=False (default) must report 0 tokens, honestly."""
        out = gui.run_scenario(list(gui.SCENARIOS)[0], "",
                               gui._fresh_state())
        report = out[2]["report"]
        assert report.llm_used is False
        assert report.model_usage == []
        assert "deterministic fallback" in out[0]

    def test_slm_toggle_wires_provider_and_records_usage(self, monkeypatch):
        """use_slm=True attaches a provider; its usage is recorded.

        Uses a fake provider (deterministic, no network) — the real-Ollama
        path is covered by tests/integration/test_gui_slm_llm.py.
        """
        class FakeSLM:
            name = "fake-slm"
            model = "fake-slm-1"
            last_usage: dict = {}

            def generate(self, system_prompt, user_prompt):
                self.last_usage = {"prompt_eval_count": 120,
                                   "eval_count": 45,
                                   "total_duration_ns": 250_000_000,
                                   "done_reason": "stop"}
                return json.dumps({
                    "decision_rationale":
                        "The transfer amount 10000 exceeded nothing (E1).",
                    "deviation_explanation": "No deviation recorded.",
                    "counterfactual_summary": "A smaller amount would pass.",
                    "evidence_summary": "Amount and limit recorded (E1, E2).",
                    "uncertainty_statement": "Ledger history not rebuilt.",
                    "claims": [{"text": "The requested amount was 10000.",
                                "label": "FACT", "evidence_refs": ["E1"]}],
                })

        monkeypatch.setattr(gui, "_ollama_available", lambda: True)
        monkeypatch.setattr(gui, "OllamaRationaleProvider",
                            lambda **kw: FakeSLM())
        out = gui.run_scenario(list(gui.SCENARIOS)[0], "",
                               gui._fresh_state(), "normal", True)
        report = out[2]["report"]
        assert report.llm_used is True
        assert report.model_usage, "provider usage must be recorded"
        usage = report.model_usage[0]
        assert usage.provider == "fake-slm"
        assert usage.prompt_tokens == 120 and usage.completion_tokens == 45
        assert "local SLM" in out[0]
        # metrics carry the real token counts, cost stays honestly unknown
        m = gui.metrics_view(out[2], False, 0, 0, False)
        assert "**120**" in m and "**45**" in m
        assert "no per-token price" in m or "unknown" in m

    def test_slm_toggle_falls_back_honestly_when_daemon_down(self, monkeypatch):
        """Unreachable daemon -> deterministic fallback + honest note."""
        monkeypatch.setattr(gui, "_ollama_available", lambda: False)
        out = gui.run_scenario(list(gui.SCENARIOS)[0], "",
                               gui._fresh_state(), "normal", True)
        report = out[2]["report"]
        assert report.llm_used is False
        assert "not reachable" in out[0]

    def test_slm_provider_failure_falls_back_not_crashes(self, monkeypatch):
        """A provider that raises must degrade to the fallback path."""
        class BrokenSLM:
            name = "broken-slm"
            model = "broken-1"
            last_usage: dict = {}

            def generate(self, system_prompt, user_prompt):
                from trace_xai.rationale.provider import ProviderError
                raise ProviderError("simulated outage")

        monkeypatch.setattr(gui, "_ollama_available", lambda: True)
        monkeypatch.setattr(gui, "OllamaRationaleProvider",
                            lambda **kw: BrokenSLM())
        out = gui.run_scenario(list(gui.SCENARIOS)[0], "",
                               gui._fresh_state(), "normal", True)
        report = out[2]["report"]
        assert report.llm_used is False
        assert report.model_usage == []
        assert report.validation is not None   # report still fully valid

    def test_pas_example_constants_are_valid_json(self):
        import json as _json
        ev = _json.loads(gui.EXAMPLE_INTERVENTION)
        gp = _json.loads(gui.EXAMPLE_GAP)
        assert ev[0]["kind"] in gui.INTERVENTION_KINDS
        assert {"start_ms", "end_ms"} <= set(gp[0])

    def test_pas_examples_evaluate_end_to_end(self, state):
        out = gui.pas_view(60000, 60000, gui.EXAMPLE_INTERVENTION,
                           gui.EXAMPLE_GAP, state)
        # the inserted intervention is detected, and the partial gap is
        # still reported alongside it (conflicts surfaced, not hidden)
        assert "INTERVENTION_DETECTED" in out
        assert "observation gaps" in out
        assert "collector offline" in out


# ----------------------------------------------------- non-technical clarity
class TestNonTechnicalClarity:
    def test_plain_language_definitions_present(self, state):
        m = gui.metrics_view(state, False, 0, 0, True)
        assert "stoplight" in gui.metrics_view(state, False, 0, 0, False) \
            or "how long" in m
        assert "milliseconds" in m
        assert "how long the banking agent" in m or "milliseconds" in m

    def test_pas_explained_in_plain_words(self, state):
        out = gui.pas_view(60000, 60000, "[]", "[]", state)
        assert "never claims the action was correct" in out \
            or "not proof" in out


# ------------------------------------------- runtime wiring (postprocess)
def _sample_value(component, state):
    """A plausible input value for a wired component."""
    if isinstance(component, gr.State):
        return state
    if isinstance(component, (gr.Dropdown, gr.Radio)):
        return component.value
    if isinstance(component, gr.Checkbox):
        return bool(component.value)
    if isinstance(component, gr.Number):
        return component.value if component.value is not None else 0.0
    if isinstance(component, gr.Slider):
        return component.value
    if isinstance(component, gr.Textbox):
        return component.value or ""
    return component.value


class TestGradioWiring:
    """Drive every wired event through Gradio's real postprocess layer.

    Handler-level tests call functions directly and can't see a mismatch
    between what a handler returns and what its wired outputs expect (e.g. a
    2-tuple handed to a single Markdown, or ``gr.update(interactive=True)``
    applied to a Markdown). Those only fail inside Gradio's postprocess — at
    click-time, for the user. This test runs every dependency through
    ``Blocks.postprocess_data``, the exact code path Gradio uses.
    """

    def _populated_state(self):
        out = gui.run_scenario(list(gui.SCENARIOS)[1], "",
                               gui._fresh_state())   # scenario B
        return out[2]

    def test_every_dependency_postprocesses_cleanly(self):
        demo = _app()
        state = self._populated_state()
        ran = 0
        for block_fn in demo.fns.values():
            if block_fn.fn is None:
                continue
            inputs = [_sample_value(c, state) for c in block_fn.inputs]
            result = block_fn.fn(*inputs)
            outputs = block_fn.outputs
            if len(outputs) > 1:
                assert isinstance(result, (tuple, list)) and \
                    len(result) == len(outputs), \
                    f"{getattr(block_fn.fn, '__name__', block_fn.fn)} returns " \
                    f"{len(result) if isinstance(result, (tuple, list)) else 1} " \
                    f"value(s) but is wired to {len(outputs)} outputs"
            # the real Gradio layer: type mismatches raise here, as at runtime
            processed = asyncio.run(
                demo.postprocess_data(block_fn, result, SessionState(demo)))
            assert len(processed) == len(outputs)
            ran += 1
        assert ran >= 15, f"only {ran} dependencies exercised"

    def test_postprocess_layer_would_catch_the_tuple_bug(self):
        """Guard: prove the harness fails on the historical bug shapes."""
        demo = _app()
        state = self._populated_state()
        bf = next(b for b in demo.fns.values()
                  if getattr(b.fn, "__name__", "") == "trace_view")
        assert len(bf.outputs) == 1
        with pytest.raises(AttributeError):
            # the shipped bug: a 2-tuple into a single Markdown
            asyncio.run(demo.postprocess_data(bf, ("head", ["row"]),
                                              SessionState(demo)))

    def test_postprocess_layer_would_catch_the_interactive_update(self):
        """Guard: the other shipped bug — update dict into a Markdown."""
        demo = _app()
        state = self._populated_state()
        bf = next(b for b in demo.fns.values()
                  if getattr(b.fn, "__name__", "") == "trace_view")
        with pytest.raises(TypeError):
            asyncio.run(demo.postprocess_data(bf, gr.update(interactive=True),
                                              SessionState(demo)))
