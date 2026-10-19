"""Post-Action Stability tests (Part 4): bounded, deterministic, caveated.

Every test injects logical-clock timestamps — no real waiting anywhere.
Covered: intervention detection, clean windows, insufficient observation,
observation gaps/interruptions, configurable window lengths, events outside
the window, and the honesty rules (never indefinite stability, never proof
of correctness, monitoring end reason always recorded).
"""

from datetime import datetime, timedelta, timezone

import pytest

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.stability import (
    DEFAULT_WINDOW_MS,
    InterventionEvent,
    InterventionType,
    MonitoringEndReason,
    PostActionStabilityMonitor,
    StabilityStatus,
)

T0 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)


def _at(ms: float) -> datetime:
    return T0 + timedelta(milliseconds=ms)


def _event(ms: float, kind=InterventionType.HUMAN_OVERRIDE,
           description="operator intervened", source="simulated"):
    return InterventionEvent(timestamp=_at(ms), kind=kind,
                             description=description, source=source)


# ------------------------------------------------------------ happy paths
class TestInterventionDetected:
    def test_intervention_in_window_is_detected(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        monitor.observe(_event(20_000, InterventionType.ROLLBACK,
                               description="transfer reversed"))
        result = monitor.close(observe_until=_at(60_000))
        assert result.status is StabilityStatus.INTERVENTION_DETECTED
        assert result.ended_because is MonitoringEndReason.INTERVENTION_DETECTED
        assert result.intervention.kind is InterventionType.ROLLBACK
        assert result.intervention_free_duration_ms == pytest.approx(20_000)
        assert result.intervention_count == 1
        assert result.monitoring_duration_ms == pytest.approx(60_000)

    def test_first_intervention_anchors_free_duration(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        monitor.observe(_event(30_000, InterventionType.MANUAL_CORRECTION))
        monitor.observe(_event(10_000, InterventionType.ROLLBACK))
        result = monitor.close(observe_until=_at(40_000))
        assert result.intervention.kind is InterventionType.ROLLBACK
        assert result.intervention_free_duration_ms == pytest.approx(10_000)
        assert result.intervention_count == 2

    def test_all_intervention_types_accepted(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        for i, kind in enumerate(InterventionType):
            monitor.observe(_event(1_000 * (i + 1), kind))
        result = monitor.close(observe_until=_at(60_000))
        assert result.intervention_count == len(list(InterventionType))


# ------------------------------------------------------------ clean window
class TestNoInterventionObserved:
    def test_full_clean_window(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        result = monitor.close(observe_until=_at(60_000))
        assert result.status is StabilityStatus.NO_INTERVENTION_OBSERVED
        assert result.ended_because is MonitoringEndReason.WINDOW_ELAPSED
        assert result.intervention is None
        assert result.intervention_free_duration_ms is None  # not invented
        assert result.monitoring_duration_ms == pytest.approx(60_000)

    def test_clean_window_never_claims_correctness(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        result = monitor.close(observe_until=_at(60_000))
        text = result.to_text().lower()
        assert "not proof" in text
        assert "bounded" in text
        assert any("not proof that the action was correct" in l
                   for l in result.limitations)

    def test_observing_beyond_window_stays_bounded(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        monitor.observe(_event(120_000))          # after the window
        result = monitor.close(observe_until=_at(120_000))
        assert result.status is StabilityStatus.NO_INTERVENTION_OBSERVED
        assert result.intervention_count == 0     # late event excluded
        assert any("after the window end" in n for n in result.notes)

    def test_event_before_action_excluded(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        early = InterventionEvent(timestamp=_at(-5_000),
                                  kind=InterventionType.OTHER)
        monitor.observe(early)
        result = monitor.close(observe_until=_at(60_000))
        assert result.status is StabilityStatus.NO_INTERVENTION_OBSERVED
        assert any("predate action completion" in n for n in result.notes)


# -------------------------------------------------- insufficient observation
class TestInsufficientObservation:
    def test_short_observation_is_insufficient(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        result = monitor.close(observe_until=_at(20_000))
        assert result.status is StabilityStatus.INSUFFICIENT_OBSERVATION
        assert result.ended_because is MonitoringEndReason.OBSERVATION_CLOSED
        assert any("shorter than" in l for l in result.limitations)

    def test_gap_covering_window_forces_insufficient(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        monitor.record_gap(_at(0), _at(60_000),
                           reason="collector offline (simulated outage)")
        result = monitor.close(observe_until=_at(60_000))
        assert result.status is StabilityStatus.INSUFFICIENT_OBSERVATION
        assert result.observation_gaps
        assert any("observation gap" in l for l in result.limitations)

    def test_partial_gap_still_insufficient(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        monitor.record_gap(_at(10_000), _at(50_000), reason="network loss")
        result = monitor.close(observe_until=_at(60_000))
        assert result.status is StabilityStatus.INSUFFICIENT_OBSERVATION
        assert result.observable_duration_ms == pytest.approx(20_000)

    def test_gap_duration_recorded(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        monitor.record_gap(_at(5_000), _at(15_000), reason="simulated gap")
        result = monitor.close(observe_until=_at(60_000))
        gap = result.observation_gaps[0]
        assert gap.duration_ms == pytest.approx(10_000)
        assert result.monitoring_duration_ms == pytest.approx(60_000)

    def test_intervention_during_gap_is_not_silently_promoted(self):
        # an intervention whose timestamp falls inside a gap is still
        # reported (the monitor does not drop data), but the observation
        # record remains explicit about the gap
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        monitor.record_gap(_at(5_000), _at(15_000), reason="missing data")
        monitor.observe(_event(10_000, InterventionType.SYSTEM_RESTART))
        result = monitor.close(observe_until=_at(60_000))
        assert result.status is StabilityStatus.INTERVENTION_DETECTED
        assert result.observation_gaps, "the gap must still be disclosed"


# ------------------------------------------------- configurable windows
class TestConfigurableWindow:
    @pytest.mark.parametrize("window_ms", [1_000, 30_000, 300_000, 3_600_000])
    def test_window_length_is_configurable(self, window_ms):
        monitor = PostActionStabilityMonitor(T0, window_ms=window_ms)
        assert monitor.window_ms == window_ms
        result = monitor.close(observe_until=_at(window_ms))
        assert result.monitoring_window_ms == window_ms
        assert result.status is StabilityStatus.NO_INTERVENTION_OBSERVED

    def test_default_window_documented(self):
        monitor = PostActionStabilityMonitor(T0)
        assert monitor.window_ms == DEFAULT_WINDOW_MS

    def test_same_event_different_windows(self):
        # an intervention at +45 s is in a 60 s window but after a 30 s one
        m60 = PostActionStabilityMonitor(T0, window_ms=60_000)
        m60.observe(_event(45_000))
        r60 = m60.close(observe_until=_at(60_000))
        m30 = PostActionStabilityMonitor(T0, window_ms=30_000)
        m30.observe(_event(45_000))
        r30 = m30.close(observe_until=_at(60_000))
        assert r60.status is StabilityStatus.INTERVENTION_DETECTED
        assert r30.status is StabilityStatus.NO_INTERVENTION_OBSERVED

    def test_nonpositive_window_rejected(self):
        with pytest.raises(ValueError):
            PostActionStabilityMonitor(T0, window_ms=0)


# ------------------------------------------------------------ input errors
class TestInputValidation:
    def test_observe_until_before_action_rejected(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        with pytest.raises(ValueError):
            monitor.close(observe_until=_at(-1_000))

    def test_reversed_gap_rejected(self):
        monitor = PostActionStabilityMonitor(T0, window_ms=60_000)
        with pytest.raises(ValueError):
            monitor.record_gap(_at(10_000), _at(5_000), reason="backwards")


# ------------------------------------------------------- deterministic reruns
class TestDeterminism:
    def test_identical_feeds_identical_results(self):
        def build():
            m = PostActionStabilityMonitor(T0, window_ms=60_000)
            m.observe(_event(12_000, InterventionType.ROLLBACK))
            m.record_gap(_at(30_000), _at(32_000), reason="simulated gap")
            return m.close(observe_until=_at(60_000)).model_dump()

        assert build() == build()


# ------------------------------------------------------- seeded from a trace
class TestSeededFromTrace:
    def test_monitor_from_pipeline_trace(self):
        """The monitor anchors on the trace's own logical-clock timestamps."""
        pipeline = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()))
        _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        trace = pipeline._traces[report.task_id]
        anchor = trace.events[-1].timestamp
        assert anchor is not None, "recorder provides logical-clock stamps"

        monitor = PostActionStabilityMonitor(anchor, window_ms=60_000)
        result = monitor.close(observe_until=anchor + timedelta(seconds=60))
        assert result.status is StabilityStatus.NO_INTERVENTION_OBSERVED
        assert result.synthetic is True
        assert any("synthetic" in l for l in result.limitations)

    def test_later_intervention_on_trace_timeline(self):
        pipeline = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()))
        _, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
        trace = pipeline._traces[report.task_id]
        anchor = trace.events[-1].timestamp

        monitor = PostActionStabilityMonitor(anchor, window_ms=60_000)
        monitor.observe(InterventionEvent(
            timestamp=anchor + timedelta(seconds=25),
            kind=InterventionType.HUMAN_OVERRIDE,
            description="compliance officer reversed the blocked attempt",
            source="simulated"))
        result = monitor.close(observe_until=anchor + timedelta(seconds=60))
        assert result.status is StabilityStatus.INTERVENTION_DETECTED
        assert result.intervention_free_duration_ms == pytest.approx(25_000)
        assert "simulated" in result.sources
