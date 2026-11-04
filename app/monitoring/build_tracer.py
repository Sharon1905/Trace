"""Build tracing for phase-4.2 build performance: stage-based durations, outcomes, errors."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


@dataclass
class BuildStage:
    """Single stage of a build operation."""

    name: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: str = "pending"
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _started_perf: float = field(default_factory=time.perf_counter, repr=False, compare=False)
    _completed_perf: Optional[float] = field(default=None, repr=False, compare=False)

    @property
    def duration_ms(self) -> float:
        """Total duration for this stage (high-resolution monotonic clock)."""
        start = self._started_perf
        end = self._completed_perf
        if end is not None:
            return (end - start) * 1000
        return 0.0


@dataclass
class BuildOutcome:
    """Outcome of a build operation."""

    build_id: str
    project_id: str
    trigger: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: str = "pending"
    error: Optional[str] = None

    total_stages: int = 0
    stages_completed: int = 0
    stages_failed: int = 0
    stage_times_ms: list[float] = field(default_factory=list)
    exceptions_raised: int = 0
    stages: Optional[list[BuildStage]] = None
    _started_perf: float = field(default_factory=time.perf_counter, repr=False, compare=False)
    _completed_perf: Optional[float] = field(default=None, repr=False, compare=False)

    @property
    def duration_ms(self) -> float:
        """Total duration for the build (high-resolution monotonic clock)."""
        start = self._started_perf
        end = self._completed_perf
        if end is not None:
            return (end - start) * 1000
        return 0.0

    @property
    def success(self) -> bool:
        """Whether the build completed successfully."""
        return self.status == "completed" and self.stages_failed == 0


class BuildTracer:
    """Traces build operations with per-stage timing, outcomes, and errors."""

    def __init__(
        self,
        max_builds: int = 500,
        max_stages_per_build: int = 50,
        record_on_start: bool = True,
    ):
        """Initialize build tracer.

        Args:
            max_builds: Maximum number of builds to keep history
            max_stages_per_build: Maximum number of stages per build
            record_on_start: Record builds on start (instead of on completion)
        """
        self.max_builds = max_builds
        self.max_stages_per_build = max_stages_per_build
        self.record_on_start = record_on_start

        self._builds: dict[str, BuildOutcome] = {}
        self._stages: dict[str, list[BuildStage]] = {}
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[BuildOutcome], None]] = []

    def start_build(
        self,
        build_id: str,
        project_id: str,
        trigger: str,
        total_stages: int = 0,
    ) -> BuildOutcome:
        """Start tracking a new build.

        Args:
            build_id: Unique identifier for this build
            project_id: Project being built/indexed
            trigger: How the build was triggered
            total_stages: Estimated total number of stages

        Returns:
            The build outcome tracking object
        """
        with self._lock:
            build = BuildOutcome(
                build_id=build_id,
                project_id=project_id,
                trigger=trigger,
                total_stages=total_stages,
            )
            self._builds[build_id] = build
            self._stages[build_id] = []
        return build

    def start_stage(self, build_id: str, stage_name: str) -> BuildStage:
        """Start a build stage.

        Args:
            build_id: Build being tracked
            stage_name: Name of the stage (e.g., "parse", "chunk", "embed")

        Returns:
            The stage tracking object
        """
        with self._lock:
            if build_id not in self._stages:
                return BuildStage(name=stage_name)

            stage = BuildStage(name=stage_name)
            stages = self._stages[build_id]
            stages.append(stage)
            build = self._builds[build_id]
            build.total_stages = len(stages)
            build.stages = stages
            return stage

    def complete_stage(
        self,
        build_id: str,
        stage_name: str,
        status: str = "completed",
        error: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> Optional[BuildStage]:
        """Complete a build stage.

        Args:
            build_id: Build being tracked
            stage_name: Name of the stage
            status: Stage status (completed/failed)
            error: Error message if failed
            meta: Additional metadata

        Returns:
            The completed stage if found, None otherwise
        """
        with self._lock:
            if build_id not in self._stages:
                return None

            stages = self._stages[build_id]
            # Find the most recent unmatched stage with this name
            for stage in reversed(stages):
                if stage.name == stage_name and stage.completed_at is None:
                    stage.completed_at = datetime.now(timezone.utc)
                    stage._completed_perf = time.perf_counter()
                    stage.status = status
                    stage.error = error
                    if meta:
                        stage.metadata.update(meta)

                    build = self._builds[build_id]
                    build.stage_times_ms.append(stage.duration_ms)

                    if status == "failed":
                        build.stages_failed += 1
                    else:
                        build.stages_completed += 1
                    return stage

            return None

    def complete_build(
        self,
        build_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> Optional[BuildOutcome]:
        """Complete a build.

        Args:
            build_id: Build being tracked
            status: Build status (completed/failed/cancelled)
            error: Error message if failed

        Returns:
            The completed build if found, None otherwise
        """
        with self._lock:
            if build_id not in self._builds:
                return None

            build = self._builds[build_id]
            build.completed_at = datetime.now(timezone.utc)
            build._completed_perf = time.perf_counter()
            build.status = status
            if error is None and status == "failed":
                # Propagate the error from the most recent failed stage
                stages = self._stages.get(build_id, [])
                for stage in reversed(stages):
                    if stage.status == "failed" and stage.error:
                        error = stage.error
                        break
            build.error = error
            build.stages = None  # cleaned up from the outcome object

            # Notify callbacks exactly once (on completion only)
            self._notify_callbacks(build)

            # Rotate if over limit
            if len(self._builds) > self.max_builds:
                # Keep newest builds
                ordered = sorted(
                    self._builds.items(),
                    key=lambda item: item[1].started_at,
                    reverse=False,  # Oldest first
                )
                to_expire = ordered[:-self.max_builds]
                for bid, b in to_expire:
                    self._builds.pop(bid, None)
                    self._stages.pop(bid, None)

            return build

    def add_callback(self, callback: Callable[[BuildOutcome], None]) -> None:
        """Add a callback to be notified on build completion.

        Args:
            callback: Function to call with build outcome
        """
        with self._lock:
            self._callbacks.append(callback)

    def _notify_callbacks(self, build: BuildOutcome) -> None:
        """Notify all callbacks of a completed build."""
        for callback in self._callbacks:
            try:
                callback(build)
            except Exception:
                # Callback failures shouldn't kill the tracer
                pass

    def get_build(self, build_id: str) -> Optional[BuildOutcome]:
        """Get a specific build by ID.

        Args:
            build_id: Build identifier

        Returns:
            Build outcome or None if not found
        """
        with self._lock:
            return self._builds.get(build_id)

    def get_builds(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[BuildOutcome]:
        """Get recent builds, optionally filtered.

        Args:
            project_id: Filter by project
            status: Filter by status
            limit: Maximum number of builds to return

        Returns:
            List of builds
        """
        with self._lock:
            builds = list(self._builds.values())

            if project_id:
                builds = [b for b in builds if b.project_id == project_id]

            if status:
                builds = [b for b in builds if b.status == status]

            # Sort by started_at descending (newest first)
            builds.sort(key=lambda b: b.started_at, reverse=True)

            return builds[:limit]

    def get_stages(self, build_id: str) -> list[BuildStage]:
        """Get all stages for a build.

        Args:
            build_id: Build identifier

        Returns:
            List of stages
        """
        with self._lock:
            return self._stages.get(build_id, []).copy()

    def get_build_summary(self, build_id: str) -> Optional[dict]:
        """Get build summary including all metrics.

        Args:
            build_id: Build identifier

        Returns:
            Dict with build summary or None
        """
        build = self.get_build(build_id)
        if not build:
            return None

        return {
            **build.__dict__,
            "stages": [s.__dict__ for s in self.get_stages(build_id)],
        }

    def shutdown(self) -> None:
        """Shutdown this tracer instance."""
        with self._lock:
            self._builds.clear()
            self._stages.clear()
            self._callbacks.clear()


# Global build tracer
_build_tracer: Optional[BuildTracer] = None


def shutdown_build_tracer() -> None:
    """Shutdown the global build tracer."""
    global _build_tracer
    if _build_tracer:
        _build_tracer.shutdown()
        _build_tracer = None


def get_build_tracer() -> BuildTracer:
    """Get the global build tracer."""
    global _build_tracer
    if _build_tracer is None:
        _build_tracer = BuildTracer()
    return _build_tracer


def set_build_tracer(tracer: BuildTracer) -> None:
    """Set the global build tracer."""
    global _build_tracer
    _build_tracer = tracer