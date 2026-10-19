"""Post-Action Stability monitor: observe a bounded window after an action.

Design contract (research brief, Part 4):

- Explicit monitoring window and explicit intervention types — both
  configurable per run.
- Statuses: INTERVENTION_DETECTED / NO_INTERVENTION_OBSERVED /
  INSUFFICIENT_OBSERVATION. A clean window is never upgraded to a claim of
  correctness or of stability beyond the window.
- Monitoring duration and the reason monitoring ended are always recorded.
- All time is injected (``action_completed_at``, ``observe_until``,
  intervention timestamps, gap intervals) — the monitor never reads a wall
  clock, so every test is deterministic and instant.
- Interruptions/missing data become observation gaps; a gap covering the
  unobserved remainder forces INSUFFICIENT_OBSERVATION and a limitation.

Typical use::

    monitor = PostActionStabilityMonitor(
        action_completed_at=last_event.timestamp,
        window_ms=300_000)
    monitor.observe(InterventionEvent(
        timestamp=t, kind=InterventionType.HUMAN_OVERRIDE,
        description="operator reversed the transfer"))
    result = monitor.close(observe_until=t_end)
"""

from __future__ import annotations

from datetime import datetime

from .models import (
    InterventionEvent,
    MonitoringEndReason,
    ObservationGap,
    PostActionStabilityResult,
    StabilityStatus,
)

__all__ = ["PostActionStabilityMonitor"]

DEFAULT_WINDOW_MS = 300_000.0  # 5 minutes; explicit, overridable per run


class PostActionStabilityMonitor:
    """Collects observations for one action and renders one bounded verdict.

    ``action_completed_at`` anchors the window (logical-clock time, usually
    the last trace event's timestamp). ``observe_until`` at ``close()`` is
    how far observation data actually reaches; the span between gaps and
    ``observe_until`` is what the verdict may speak about.
    """

    def __init__(self, action_completed_at: datetime,
                 window_ms: float = DEFAULT_WINDOW_MS) -> None:
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        self._start = action_completed_at
        self._window_ms = float(window_ms)
        self._interventions: list[InterventionEvent] = []
        self._gaps: list[ObservationGap] = []

    # ------------------------------------------------------------ config
    @property
    def window_ms(self) -> float:
        return self._window_ms

    # ---------------------------------------------------------- feeding
    def observe(self, event: InterventionEvent) -> None:
        """Record one intervention event (timestamp injected by the caller).

        Events before the action or after the window are kept but do not
        change the verdict — the verdict speaks only about the window.
        """
        self._interventions.append(event)

    def record_gap(self, start: datetime, end: datetime,
                   reason: str) -> None:
        """Record an observation interruption / missing-data interval."""
        if end < start:
            raise ValueError("gap end must not precede start")
        self._gaps.append(ObservationGap(start=start, end=end, reason=reason))

    # ---------------------------------------------------------- verdict
    def close(self, observe_until: datetime) -> PostActionStabilityResult:
        """End monitoring at an injected time and compute the verdict."""
        if observe_until < self._start:
            raise ValueError("observe_until precedes action completion")
        window_end = _add_ms(self._start, self._window_ms)
        span_ms = (observe_until - self._start).total_seconds() * 1000.0

        in_window = [e for e in self._interventions
                     if self._start <= e.timestamp <= window_end]
        in_window.sort(key=lambda e: e.timestamp)
        gaps = [g for g in self._gaps
                if g.end > self._start and g.start < observe_until]

        observable_ms = max(0.0, span_ms - _gap_ms_within(gaps, self._start,
                                                          observe_until))
        fully_observed = (span_ms >= self._window_ms
                          and _gap_ms_within(gaps, self._start, window_end) <= 0.0)

        limitations: list[str] = [
            "observation is bounded by the configured monitoring window; "
            "no claim is made about times after the window or beyond the "
            "observed span",
            "NO_INTERVENTION_OBSERVED means no intervention was seen in the "
            "observed data — it is not proof that the action was correct",
            "all observation data is synthetic (simulated interventions), "
            "not real production signals",
        ]
        notes: list[str] = [
            f"monitoring window: {self._window_ms:.0f} ms "
            f"(configurable per run)",
            f"observed span: {span_ms:.1f} ms; observable time: "
            f"{observable_ms:.1f} ms",
        ]

        if in_window:
            first = in_window[0]
            free_ms = (first.timestamp - self._start).total_seconds() * 1000.0
            status = StabilityStatus.INTERVENTION_DETECTED
            ended = MonitoringEndReason.INTERVENTION_DETECTED
            later = len(in_window) - 1
            if later:
                notes.append(f"monitoring stopped at the first intervention; "
                             f"{later} later in-window event(s) also counted")
        else:
            free_ms = None
            if fully_observed and span_ms >= self._window_ms:
                status = StabilityStatus.NO_INTERVENTION_OBSERVED
                ended = MonitoringEndReason.WINDOW_ELAPSED
            else:
                status = StabilityStatus.INSUFFICIENT_OBSERVATION
                ended = MonitoringEndReason.OBSERVATION_CLOSED
                if span_ms < self._window_ms:
                    limitations.append(
                        f"observed span ({span_ms:.0f} ms) is shorter than "
                        f"the monitoring window ({self._window_ms:.0f} ms): "
                        f"INSUFFICIENT_OBSERVATION")
                if gaps:
                    limitations.append(
                        f"{len(gaps)} observation gap(s) interrupt the "
                        f"window; unobserved time is not evidence of "
                        f"stability")

        late = [e for e in self._interventions if e.timestamp > window_end]
        before = [e for e in self._interventions if e.timestamp < self._start]
        if late:
            notes.append(f"{len(late)} intervention(s) occurred after the "
                         f"window end and are excluded from this verdict")
        if before:
            notes.append(f"{len(before)} event(s) predate action completion "
                         f"and are excluded from this verdict")

        return PostActionStabilityResult(
            status=status,
            ended_because=ended,
            monitoring_window_ms=self._window_ms,
            monitoring_duration_ms=round(span_ms, 3),
            observable_duration_ms=round(observable_ms, 3),
            intervention_free_duration_ms=(round(free_ms, 3)
                                           if free_ms is not None else None),
            intervention=in_window[0] if in_window else None,
            intervention_count=len(in_window),
            observation_gaps=gaps,
            sources=sorted({e.source for e in in_window}),
            limitations=limitations,
            notes=notes,
        )


def _add_ms(moment: datetime, ms: float) -> datetime:
    from datetime import timedelta

    return moment + timedelta(milliseconds=ms)


def _gap_ms_within(gaps: list[ObservationGap], start: datetime,
                   end: datetime) -> float:
    """Total gap milliseconds overlapping [start, end] (clipped, merged)."""
    total = 0.0
    for g in gaps:
        lo = max(g.start, start)
        hi = min(g.end, end)
        if hi > lo:
            total += (hi - lo).total_seconds() * 1000.0
    return total
