"""Metrics collection for rebuild operations and system performance."""

from __future__ import annotations

import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class RebuildMetric:
    """Metric for a single rebuild operation."""

    job_id: str
    project_id: str
    trigger: str
    status: str
    duration_ms: float
    chunks_before: int
    chunks_after: int
    incremental: bool
    changed_files_count: int
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "trigger": self.trigger,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "chunks_before": self.chunks_before,
            "chunks_after": self.chunks_after,
            "incremental": self.incremental,
            "changed_files_count": self.changed_files_count,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CacheMetric:
    """Metric for cache operations."""

    operation: str  # hit, miss, eviction, invalidation
    project_id: Optional[str]
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "project_id": self.project_id,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class MetricsCollector:
    """Collects and aggregates system metrics."""

    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self._start_time = time.time()
        self._rebuild_metrics: deque[RebuildMetric] = deque(maxlen=max_history)
        self._cache_metrics: deque[CacheMetric] = deque(maxlen=max_history)
        self._lock = threading.RLock()

        # Aggregated counters
        self._counters: dict[str, int] = defaultdict(int)
        self._timers: dict[str, list[float]] = defaultdict(list)

    def record_rebuild(self, metric: RebuildMetric) -> None:
        """Record a rebuild operation metric."""
        with self._lock:
            self._rebuild_metrics.append(metric)
            self._counters[f"rebuild_{metric.status}"] += 1
            self._counters["rebuild_total"] += 1
            self._timers["rebuild_duration"].append(metric.duration_ms)

    def record_cache_operation(
        self,
        operation: str,
        project_id: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a cache operation metric."""
        with self._lock:
            metric = CacheMetric(
                operation=operation,
                project_id=project_id,
                latency_ms=latency_ms,
            )
            self._cache_metrics.append(metric)
            self._counters[f"cache_{operation}"] += 1
            self._timers[f"cache_{operation}_latency"].append(latency_ms)

    def record_timer(self, name: str, duration_ms: float) -> None:
        """Record a generic timer metric."""
        with self._lock:
            self._timers[name].append(duration_ms)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value

    def get_rebuild_stats(self) -> dict:
        """Get aggregated rebuild statistics."""
        with self._lock:
            if not self._rebuild_metrics:
                return {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "avg_duration_ms": 0,
                    "p50_duration_ms": 0,
                    "p95_duration_ms": 0,
                    "p99_duration_ms": 0,
                }

            completed = [m for m in self._rebuild_metrics if m.status == "completed"]
            failed = [m for m in self._rebuild_metrics if m.status == "failed"]
            durations = [m.duration_ms for m in self._rebuild_metrics]

            return {
                "total": len(self._rebuild_metrics),
                "completed": len(completed),
                "failed": len(failed),
                "success_rate": len(completed) / len(self._rebuild_metrics) if self._rebuild_metrics else 0,
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "p50_duration_ms": self._percentile(durations, 50),
                "p95_duration_ms": self._percentile(durations, 95),
                "p99_duration_ms": self._percentile(durations, 99),
                "incremental_count": sum(1 for m in self._rebuild_metrics if m.incremental),
                "full_rebuild_count": sum(1 for m in self._rebuild_metrics if not m.incremental),
                "avg_chunks_processed": sum(m.chunks_after for m in completed) / len(completed) if completed else 0,
            }

    def get_cache_stats(self) -> dict:
        """Get aggregated cache statistics."""
        with self._lock:
            hits = self._counters.get("cache_hit", 0)
            misses = self._counters.get("cache_miss", 0)
            total = hits + misses

            hit_latencies = self._timers.get("cache_hit_latency", [])
            miss_latencies = self._timers.get("cache_miss_latency", [])

            return {
                "hits": hits,
                "misses": misses,
                "total_requests": total,
                "hit_rate": hits / total if total > 0 else 0,
                "evictions": self._counters.get("cache_eviction", 0),
                "invalidations": self._counters.get("cache_invalidation", 0),
                "avg_hit_latency_ms": sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0,
                "avg_miss_latency_ms": sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0,
            }

    def get_recent_rebuilds(self, limit: int = 50) -> list[dict]:
        """Get recent rebuild metrics."""
        with self._lock:
            return [m.to_dict() for m in list(self._rebuild_metrics)[-limit:]]

    def get_recent_cache_ops(self, limit: int = 50) -> list[dict]:
        """Get recent cache operation metrics."""
        with self._lock:
            return [m.to_dict() for m in list(self._cache_metrics)[-limit:]]

    def get_all_counters(self) -> dict:
        """Get all counters."""
        with self._lock:
            return dict(self._counters)

    def get_system_health(self) -> dict:
        """Get overall system health metrics."""
        with self._lock:
            rebuild_stats = self.get_rebuild_stats()
            cache_stats = self.get_cache_stats()

            return {
                "rebuild": rebuild_stats,
                "cache": cache_stats,
                "uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
            }

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        index = int(len(sorted_vals) * percentile / 100)
        return sorted_vals[min(index, len(sorted_vals) - 1)]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._rebuild_metrics.clear()
            self._cache_metrics.clear()
            self._counters.clear()
            self._timers.clear()


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector."""
    return _metrics_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector."""
    global _metrics_collector
    _metrics_collector = collector