"""Post-Action Stability (PAS / "After-Effects"): bounded after-monitoring.

Answers one question with explicit bounds: *after this action completed, and
within this configured window, was an intervention observed?* — reporting
INTERVENTION_DETECTED / NO_INTERVENTION_OBSERVED / INSUFFICIENT_OBSERVATION,
the monitoring duration, why monitoring ended, and every observation gap.

It never claims indefinite stability or proof of correctness; a clean window
is reported as "no intervention observed in the observed data", with the
window size stated. All timestamps are injected (logical clock), so no test
waits and every result is reproducible.

Usage::

    from trace_xai.stability import (
        InterventionEvent, InterventionType, PostActionStabilityMonitor)

    monitor = PostActionStabilityMonitor(
        action_completed_at=trace.events[-1].timestamp,
        window_ms=60_000)
    monitor.observe(InterventionEvent(
        timestamp=t, kind=InterventionType.ROLLBACK,
        description="ledger restored by operator"))
    result = monitor.close(observe_until=t_end)
    print(result.to_text())
"""

from .models import (
    InterventionEvent,
    InterventionType,
    MonitoringEndReason,
    ObservationGap,
    PostActionStabilityResult,
    StabilityStatus,
)
from .monitor import DEFAULT_WINDOW_MS, PostActionStabilityMonitor

__all__ = [
    "PostActionStabilityMonitor",
    "PostActionStabilityResult",
    "InterventionEvent",
    "InterventionType",
    "MonitoringEndReason",
    "ObservationGap",
    "StabilityStatus",
    "DEFAULT_WINDOW_MS",
]
