"""Experiment regression tests (Part 6): A-E must stay green and honest.

The experiments are deterministic (deterministic fallback rationale,
injected timestamps), so they can run in the fast suite. These tests pin
each experiment's key claim and its honesty disclosures.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "final_phase_experiments",
    _ROOT / "experiments" / "final_phase_experiments.py")
experiments = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiments)


@pytest.fixture(scope="module")
def traces():
    pipeline = experiments._pipeline()
    return {name: pipeline.run(task)[1]
            for name, task in experiments.SCENARIOS.items()}


class TestExperimentA:
    def test_invariants_hold(self, traces):
        result = experiments.experiment_a(traces)
        assert result["all_invariants_hold"] is True
        assert result["unknown_violations"] == 0
        assert result["wording_varies"] is True
        assert result["sample_size"]["mode_views"] == len(traces) * 4

    def test_no_human_evaluation_implied(self, traces):
        result = experiments.experiment_a(traces)
        assert any("no human subjects" in l for l in result["limitations"])


class TestExperimentB:
    @pytest.fixture(scope="class")
    def result(self):
        return experiments.experiment_b()

    def test_full_pipeline_beats_baseline(self, result):
        assert result["baseline"]["false_acceptance_rate"] == 1.0
        full = result["full_pipeline"]
        assert full["false_acceptance_rate"] == 0.0
        assert full["detection_rate"] == 1.0
        assert full["wrong_detected"] == full["wrong_claims"] == 3
        assert result["improvement"]["false_acceptance_rate_delta"] == 1.0

    def test_invented_claim_quarantined(self, result):
        assert result["invented_claim_quarantined"] is True

    def test_ground_truth_is_not_an_llm_judge(self, result):
        assert any("no LLM judge" in l for l in result["limitations"])
        assert any("known by construction" in l for l in result["limitations"])


class TestExperimentC:
    def test_metrics_cover_all_scenarios(self):
        result = experiments.experiment_c()
        assert result["sample_size"]["scenarios"] == 5
        names = {r["scenario"] for r in result["rows"]}
        assert {"A_success", "B_violation", "D_both",
                "F_missing_evidence", "G_conflicting_evidence"} <= names
        assert result["units"]["latency"] == "ms"
        # deterministic fallback: no model consulted, cost 0 by construction
        for row in result["rows"]:
            if row["scenario"] in ("A_success", "B_violation", "D_both"):
                assert row["cost_basis"] == "measured"
                assert row["estimated_cost_usd"] == 0.0
                assert row["csat_status"] == "NOT_COLLECTED"

    def test_unknown_split_visible_for_missing_evidence(self):
        result = experiments.experiment_c()
        row = next(r for r in result["rows"]
                   if r["scenario"] == "F_missing_evidence")
        assert row["unknown_requirements"] > 0
        assert row["unverifiable_rate"] >= 0.0


class TestExperimentD:
    def test_all_statuses_as_expected(self):
        result = experiments.experiment_d()
        assert result["all_statuses_as_expected"] is True
        assert result["sample_size"]["cases"] == 7
        assert result["late_event_excluded"] is True

    def test_no_waiting_used(self):
        # the module never sleeps or reads a wall clock; it is deterministic
        result = experiments.experiment_d()
        assert any("injected clock" in l and "simulated" in l
                   for l in result["limitations"])

    def test_insufficient_cases_carry_gaps_or_short_span(self):
        result = experiments.experiment_d()
        for case in result["cases"]:
            if case["status"] == "INSUFFICIENT_OBSERVATION":
                assert (case["gap_count"] > 0
                        or case["observe_ms"] < case["window_ms"])


class TestExperimentE:
    def test_backwards_compatible(self):
        result = experiments.experiment_e()
        assert result["backwards_compatible"] is True
        for row in result["rows"]:
            assert row["core_verdicts_identical_across_views"] is True
            assert row["extended_artifacts"]["presentation_modes"] == 4
            assert row["phase8_report_sections"] == 15

    def test_phase8_report_unchanged_by_extension(self, traces):
        # the Phase 8 artifact still carries its 15 numbered sections
        report = next(iter(traces.values()))
        text = report.to_text()
        for i, title in enumerate(
                ["Task", "Requirements", "Agent Action", "Decision Rationale",
                 "Requirement Satisfaction", "Deviations",
                 "Trajectory Anomalies", "Counterfactual Alternatives",
                 "Evidence", "Validation Results", "Supported Claims",
                 "Contradicted Claims", "Unknown Claims",
                 "Confidence / Uncertainty", "Limitations"], start=1):
            assert f"[{i}] {title}" in text


class TestResultsArtifact:
    def test_results_json_is_written_and_valid(self, tmp_path):
        import json
        pipeline = experiments._pipeline()
        traces = {name: pipeline.run(task)[1]
                  for name, task in experiments.SCENARIOS.items()}
        results = [experiments.experiment_a(traces), experiments.experiment_b(),
                   experiments.experiment_c(), experiments.experiment_d(),
                   experiments.experiment_e()]
        payload = json.loads(json.dumps(results, default=str))
        assert [r["experiment"] for r in payload] == [
            "A_adaptive_presentation_consistency",
            "B_evidence_validation_baseline_vs_full",
            "C_metric_evaluation",
            "D_post_action_stability",
            "E_phase8_vs_extended",
        ]
        for r in payload:
            assert r["limitations"], f"{r['experiment']} must document limits"
            assert r["sample_size"]
