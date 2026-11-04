"""Query cache with invalidation on index rebuilds."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.schemas import AskRequest


@dataclass
class CacheEntry:
    """A cached query result."""

    key: str
    project_id: str
    question: str
    retrieval_mode: str
    response: dict
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ttl_seconds: int = 3600  # Default 1 hour TTL

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age >= self.ttl_seconds

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "project_id": self.project_id,
            "question": self.question,
            "retrieval_mode": self.retrieval_mode,
            "response": self.response,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds,
            "expired": self.is_expired(),
        }


class QueryCache:
    """Thread-safe query cache with LRU eviction and project-based invalidation."""

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 3600,
        cleanup_interval: int = 300,
    ):
        """Initialize query cache.

        Args:
            max_entries: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Interval for automatic cleanup of expired entries
        """
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._project_index: dict[str, set[str]] = {}  # project_id -> set of cache keys
        self._lock = threading.RLock()

        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
        }

        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup()

    def _start_cleanup(self) -> None:
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="query-cache-cleanup")
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Background loop to clean expired entries."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            self._cleanup_expired()

    def _cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
            return len(expired_keys)

    def _generate_key(self, request: AskRequest) -> str:
        """Generate cache key from request."""
        # Include project_id, question, retrieval_mode in key
        key_data = f"{request.project_id}:{request.retrieval_mode or 'hybrid'}:{request.question}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache and project index."""
        entry = self._cache.pop(key, None)
        if entry:
            project_keys = self._project_index.get(entry.project_id, set())
            project_keys.discard(key)
            if not project_keys:
                self._project_index.pop(entry.project_id, None)
            return True
        return False

    def get(self, request: AskRequest) -> Optional[dict]:
        """Get cached response for request."""
        key = self._generate_key(request)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None

            # Update access time and move to end (LRU)
            entry.accessed_at = datetime.now(timezone.utc)
            entry.access_count += 1
            self._cache.move_to_end(key)

            self._stats["hits"] += 1
            return entry.response

    def set(self, request: AskRequest, response: dict, ttl: Optional[int] = None) -> None:
        """Cache a response for a request."""
        key = self._generate_key(request)

        with self._lock:
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self.max_entries:
                self._evict_lru()

            entry = CacheEntry(
                key=key,
                project_id=request.project_id,
                question=request.question,
                retrieval_mode=request.retrieval_mode or "hybrid",
                response=response,
                ttl_seconds=ttl or self.default_ttl,
            )

            self._cache[key] = entry
            self._project_index.setdefault(request.project_id, set()).add(key)

    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False
        key, _ = self._cache.popitem(last=False)
        self._remove_entry(key)
        self._stats["evictions"] += 1
        return True

    def invalidate_project(self, project_id: str) -> int:
        """Invalidate all cache entries for a project.

        Args:
            project_id: Project to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys = self._project_index.get(project_id, set()).copy()
            count = 0
            for key in keys:
                if self._remove_entry(key):
                    count += 1
            self._stats["invalidations"] += count
            return count

    def invalidate_all(self) -> int:
        """Invalidate all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._project_index.clear()
            self._stats["invalidations"] += count
            return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            return {
                **self._stats,
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "projects_tracked": len(self._project_index),
                "hit_rate": hit_rate,
                "memory_estimate_mb": self._estimate_memory() / (1024 * 1024),
            }

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        total = 0
        for entry in self._cache.values():
            resp_size = len(entry.response) if isinstance(entry.response, str) else len(str(entry.response))
            total += len(entry.question) + resp_size + 200
        return total

    def get_entries(self, project_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Get cache entries for inspection."""
        with self._lock:
            if project_id:
                keys = self._project_index.get(project_id, set())
                entries = [self._cache[k] for k in keys if k in self._cache]
            else:
                entries = list(self._cache.values())

            entries.sort(key=lambda e: e.accessed_at, reverse=True)
            return [e.to_dict() for e in entries[:limit]]

    def warm_cache(self, requests: list[AskRequest], responses: list[dict]) -> int:
        """Warm cache with pre-computed responses."""
        count = 0
        for req, resp in zip(requests, responses, strict=False):
            self.set(req, resp)
            count += 1
        return count

    def shutdown(self) -> None:
        """Shutdown cache and cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)


# Global cache instance
_query_cache: Optional[QueryCache] = None


def get_query_cache() -> Optional[QueryCache]:
    """Get the global query cache."""
    return _query_cache


def set_query_cache(cache: QueryCache) -> None:
    """Set the global query cache."""
    global _query_cache
    _query_cache = cache