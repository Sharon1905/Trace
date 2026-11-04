"""File system watcher for detecting changes in TRACE projects."""

from __future__ import annotations

import os
import threading
import time
import weakref
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    Observer = None  # type: ignore
    # Define stub class if watchdog is not available
    class FileSystemEventHandler:  # type: ignore
        pass


class FileWatcher:
    """Monitor project directories for file changes with debouncing."""

    def __init__(self, watch_path: Path, debounce_ms: int = 500, max_history: int = 100):
        """Initialize file watcher.

        Args:
            watch_path: Root directory to watch (typically data/projects/)
            debounce_ms: Milliseconds to debounce rapid changes (default 500)
            max_history: Maximum change events to keep in history (default 100)
        """
        self.watch_path = Path(watch_path)
        self.debounce_ms = debounce_ms
        self.max_history = max_history

        self.observer: Optional[Observer] = None
        self.event_handler: Optional[_ProjectEventHandler] = None
        self.changes: deque = deque(maxlen=max_history)
        self.lock = threading.Lock()
        self.running = False
        self.on_change: Optional[Callable[[list[dict]], None]] = None

    def start(self) -> None:
        """Start watching for file changes."""
        if not HAS_WATCHDOG:
            return  # Gracefully disable if watchdog unavailable

        if self.running:
            return

        try:
            self.event_handler = _ProjectEventHandler(
                self.watch_path,
                self.debounce_ms,
                self.changes,
                self.lock,
            )
            # Set weak reference for callback
            self.event_handler._watcher_ref = weakref.ref(self)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, str(self.watch_path), recursive=True)
            self.observer.start()
            self.running = True
        except Exception as e:
            # Gracefully degrade if watchdog fails
            pass

    def stop(self) -> None:
        """Stop watching for file changes."""
        if self.event_handler is not None:
            self.event_handler.cancel_all()

        if not self.running or not self.observer:
            return

        try:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.running = False
        except Exception:
            pass

    def status(self) -> dict:
        """Return current watcher status."""
        return {
            "running": self.running,
            "has_watchdog": HAS_WATCHDOG,
            "watch_path": str(self.watch_path),
            "debounce_ms": self.debounce_ms,
            "change_count": len(self.changes),
        }

    def get_changes(self) -> list[dict]:
        """Get accumulated change events since last call."""
        with self.lock:
            changes_list = list(self.changes)
            self.changes.clear()
        return changes_list

    def cancel_project(self, project_id: str) -> None:
        """Cancel pending debounce timers and changes for a project.

        Thread-safe. Safely handles:
        - A watcher that has been initialized and started.
        - A watcher that has not been started (event_handler is None).
        - A watcher that has already stopped.
        - A project with pending debounce timers.
        - A project with queued file changes.
        - A project whose directory is already missing.
        """
        if self.event_handler is not None:
            self.event_handler.cancel_project(project_id)

        with self.lock:
            retained = [c for c in self.changes if c.get("project_id") != project_id]
            self.changes.clear()
            self.changes.extend(retained)


class _ProjectEventHandler(FileSystemEventHandler):
    """Handle file system events and debounce changes."""

    def __init__(self, watch_path: Path, debounce_ms: int, changes: deque, lock: threading.Lock):
        """Initialize event handler.

        Args:
            watch_path: Root directory being watched
            debounce_ms: Debounce delay in milliseconds
            changes: Deque to accumulate changes
            lock: Thread lock for thread-safe access
        """
        super().__init__()
        self.watch_path = watch_path
        self.debounce_ms = debounce_ms / 1000.0  # Convert to seconds
        self.changes = changes
        self.lock = lock

        # Debounce state: track pending changes per project
        self.pending_changes: dict[str, dict] = {}
        self.debounce_timers: dict[str, threading.Timer] = {}
        self.cancelled_projects: set[str] = set()
        self.debounce_lock = threading.Lock()
        self._watcher_ref = None

    def cancel_project(self, project_id: str) -> None:
        """Cancel pending debounce timer and clear pending changes for a project.

        Thread-safe. Prevents in-flight debounce callbacks and upcoming events
        for this project from scheduling rebuilds.
        """
        with self.debounce_lock:
            self.cancelled_projects.add(project_id)
            timer = self.debounce_timers.pop(project_id, None)
            if timer is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
            self.pending_changes.pop(project_id, None)

    def cancel_all(self) -> None:
        """Cancel all pending debounce timers and clear all pending changes."""
        with self.debounce_lock:
            for timer in self.debounce_timers.values():
                try:
                    timer.cancel()
                except Exception:
                    pass
            self.debounce_timers.clear()
            self.pending_changes.clear()

    def on_modified(self, event) -> None:
        """Handle file modification events."""
        if not event.is_directory and self._is_python_file(event.src_path):
            self._queue_change("modified", event.src_path)

    def on_created(self, event) -> None:
        """Handle file creation events."""
        if not event.is_directory and self._is_python_file(event.src_path):
            self._queue_change("created", event.src_path)

    def on_deleted(self, event) -> None:
        """Handle file deletion events."""
        if not event.is_directory and self._is_python_file(event.src_path):
            self._queue_change("deleted", event.src_path)

    def _is_python_file(self, path: str) -> bool:
        """Check if path is a Python file to monitor."""
        path_obj = Path(path)

        # Ignore certain patterns
        parts = path_obj.parts
        if any(part in ("__pycache__", ".git", ".trace") for part in parts):
            return False

        # Only watch .py files
        return path_obj.suffix == ".py"

    def _get_project_id(self, file_path: str) -> Optional[str]:
        """Extract project ID from file path."""
        try:
            path_obj = Path(file_path)
            relative = path_obj.relative_to(self.watch_path)
            parts = relative.parts

            # Project ID should be first part (32-char hex)
            if parts and len(parts[0]) == 32:
                return parts[0]
        except (ValueError, IndexError):
            pass

        return None

    def _queue_change(self, event_type: str, file_path: str) -> None:
        """Queue a change event with debouncing."""
        project_id = self._get_project_id(file_path)
        if not project_id:
            return

        with self.debounce_lock:
            if project_id in self.cancelled_projects:
                return

            # Cancel existing timer for this project
            if project_id in self.debounce_timers:
                self.debounce_timers[project_id].cancel()

            # Update pending change
            try:
                relative_path = str(Path(file_path).relative_to(self.watch_path / project_id))
            except ValueError:
                return

            self.pending_changes[project_id] = {
                "type": "project_modified",
                "project_id": project_id,
                "event": event_type,
                "file": relative_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Set new debounce timer
            timer = threading.Timer(
                self.debounce_ms,
                self._flush_change,
                args=(project_id,),
            )
            self.debounce_timers[project_id] = timer
            timer.start()

    def _flush_change(self, project_id: str) -> None:
        """Flush a debounced change to the queue."""
        with self.debounce_lock:
            if project_id in self.cancelled_projects:
                self.pending_changes.pop(project_id, None)
                self.debounce_timers.pop(project_id, None)
                return

            if project_id in self.pending_changes:
                change = self.pending_changes.pop(project_id)
                self.debounce_timers.pop(project_id, None)

                with self.lock:
                    self.changes.append(change)

                # Trigger callback if set
                if self._watcher_ref and self._watcher_ref().on_change:
                    try:
                        self._watcher_ref().on_change([change])
                    except Exception:
                        pass
