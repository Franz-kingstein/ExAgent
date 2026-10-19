"""Post-Action Stability (PAS / "After-Effects") models.

What this module claims — and what it deliberately does not:

- It observes a *bounded* monitoring window after an action completed and
  reports whether an intervention (human override, rollback, manual
  correction, ...) was detected in that window.
- ``NO_INTERVENTION_OBSERVED`` means exactly that: nothing was seen inside
  the configured window. It is **not** proof of correctness, not a claim of
  indefinite stability, and not a guarantee beyond the observed window.
- Interruptions and missing observation data are first-class: they become
  gaps, which force ``INSUFFICIENT_OBSERVATION`` when they cover the window,
  and are always listed as limitations.
- All timestamps are injected (logical clock, same convention as the trace
  recorder). Nothing here calls the wall clock, so tests never wait.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class InterventionType(str, Enum):
    """Kinds of post-action interventions the monitor can be told about."""

    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"          # a person changed the outcome
    ROLLBACK = "ROLLBACK"                      # the action was undone
    MANUAL_CORRECTION = "MANUAL_CORRECTION"    # state fixed by hand
    POLICY_CHANGE = "POLICY_CHANGE"            # rule changed after the action
    SYSTEM_RESTART = "SYSTEM_RESTART"          # component restarted mid-window
    OTHER = "OTHER"                            # recorded, not force-classified


class StabilityStatus(str, Enum):
    INTERVENTION_DETECTED = "INTERVENTION_DETECTED"
    NO_INTERVENTION_OBSERVED = "NO_INTERVENTION_OBSERVED"
    INSUFFICIENT_OBSERVATION = "INSUFFICIENT_OBSERVATION"


class MonitoringEndReason(str, Enum):
    """Why monitoring stopped — recorded, never left implicit."""

    WINDOW_ELAPSED = "WINDOW_ELAPSED"              # full window observed
    INTERVENTION_DETECTED = "INTERVENTION_DETECTED"  # stopped at intervention
    OBSERVATION_CLOSED = "OBSERVATION_CLOSED"      # caller closed early
    DATA_ENDED = "DATA_ENDED"                      # observation data ends


class InterventionEvent(BaseModel):
    """One reported intervention inside the monitoring window."""

    timestamp: datetime = Field(description="logical-clock time of the event")
    kind: InterventionType
    description: str = ""
    source: str = Field(
        default="simulated",
        description="how this event entered the monitor (simulated | reported); "
                    "PAS in TRACE-XAI runs on synthetic data, so the default "
                    "is 'simulated' and is surfaced in the result",
    )


class ObservationGap(BaseModel):
    """A stretch of time with no observation (interruption / missing data)."""

    start: datetime
    end: datetime
    reason: str = Field(description="why observation was unavailable")

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start).total_seconds() * 1000.0


class PostActionStabilityResult(BaseModel):
    """The verdict of one monitoring run — bounded, dated and caveated."""

    status: StabilityStatus
    ended_because: MonitoringEndReason

    # window configuration and what was actually observed
    monitoring_window_ms: float = Field(gt=0)
    monitoring_duration_ms: float = Field(
        ge=0, description="observed_until - action_completed (raw span)")
    observable_duration_ms: float = Field(
        ge=0, description="raw span minus observation gaps")

    intervention_free_duration_ms: float | None = Field(
        default=None,
        description="time from action completion to the first intervention; "
                    "None when no intervention was observed",
    )
    intervention: InterventionEvent | None = None
    intervention_count: int = 0

    observation_gaps: list[ObservationGap] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list,
        description="distinct sources of observed intervention events",
    )

    limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    method: str = "post_action_stability_v1"
    synthetic: bool = True

    # ---------------------------------------------------------- rendering
    def to_text(self) -> str:
        lines = [
            "=" * 72,
            "POST-ACTION STABILITY (after-effects)",
            "=" * 72,
            f"status: {self.status.value}",
            f"ended because: {self.ended_because.value}",
            f"monitoring window: {self.monitoring_window_ms:.0f} ms",
            f"observed span: {self.monitoring_duration_ms:.1f} ms "
            f"(observable {self.observable_duration_ms:.1f} ms)",
        ]
        if self.intervention_free_duration_ms is not None:
            lines.append(f"intervention-free for: "
                         f"{self.intervention_free_duration_ms:.1f} ms")
        else:
            lines.append("intervention-free duration: none observed "
                         "(no intervention seen in the observed span)")
        if self.intervention is not None:
            i = self.intervention
            lines.append(f"first intervention: {i.kind.value} at "
                         f"{i.timestamp.isoformat()} "
                         f"({i.description or 'no description'}) "
                         f"[source: {i.source}]")
        lines.append(f"intervention count: {self.intervention_count}")
        if self.observation_gaps:
            lines.append("observation gaps:")
            for g in self.observation_gaps:
                lines.append(f"  - {g.start.isoformat()} -> "
                             f"{g.end.isoformat()} "
                             f"({g.duration_ms:.0f} ms): {g.reason}")
        else:
            lines.append("observation gaps: none")
        if self.sources:
            lines.append(f"event sources: {', '.join(self.sources)}")
        if self.limitations:
            lines.append("limitations:")
            lines.extend(f"  - {l}" for l in self.limitations)
        if self.notes:
            lines.append("notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        lines.append("-" * 72)
        lines.append("synthetic data only | window-bounded observation; "
                     "never a claim of indefinite stability")
        return "\n".join(lines)
