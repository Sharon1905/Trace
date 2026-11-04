"""Rebuild coordinator for automatic index rebuilding triggered by file watcher events."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from app.chunking.chunker import build_code_chunks
from app.embeddings.embedder import EmbeddingError, EmbeddingSettings, SentenceTransformerEmbedder
from app.ingestion.extractor import ExtractionError, extract_project
from app.monitoring import file_watcher
from app.monitoring.build_tracer import BuildTracer, get_build_tracer
from app.monitoring.file_watcher import FileWatcher
from app.parsing.ast_parser import parse_python_file
from app.storage.project_store import ProjectIndexError, ProjectIndexMissing, ProjectStore
from app.utils.project_manager import get_data_dir, get_project
from app.vectorstore.faiss_store import FAISSStore, canonical_chunks


class RebuildStatus(str, Enum):
    """Status of a rebuild job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RebuildTrigger(str, Enum):
    """What triggered the rebuild."""

    FILE_WATCHER = "file_watcher"
    MANUAL = "manual"
    STARTUP = "startup"
    WEBHOOK = "webhook"


@dataclass
class RebuildJob:
    """A rebuild job queued for execution."""

    project_id: str
    trigger: RebuildTrigger = RebuildTrigger.MANUAL
    changed_files: list[str] = field(default_factory=list)
    priority: int = 0  # Higher = more urgent
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RebuildStatus = RebuildStatus.PENDING
    error: Optional[str] = None
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    incremental: bool = False
    progress: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "trigger": self.trigger.value,
            "changed_files": self.changed_files,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "error": self.error,
            "incremental": self.incremental,
            "progress": self.progress,
        }


class RebuildCoordinator:
    """Coordinates automatic index rebuilding from file watcher events."""

    def __init__(
        self,
        watcher: Optional[FileWatcher] = None,
        max_concurrent_rebuilds: int = 1,
        enable_incremental: bool = True,
        rebuild_callback: Optional[Callable[[RebuildJob], None]] = None,
    ):
        """Initialize rebuild coordinator.

        Args:
            watcher: Optional file watcher to monitor for changes
            max_concurrent_rebuilds: Max simultaneous rebuilds (default 1)
            enable_incremental: Whether to use incremental indexing
            rebuild_callback: Optional callback when rebuild completes
        """
        self.watcher = watcher
        self.max_concurrent_rebuilds = max_concurrent_rebuilds
        self.enable_incremental = enable_incremental
        self.rebuild_callback = rebuild_callback

        self._job_queue: deque[RebuildJob] = deque()
        self._running_jobs: dict[str, RebuildJob] = {}
        self._completed_jobs: deque[RebuildJob] = deque(maxlen=100)
        self._lock = threading.RLock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Project store and FAISS store
        self.project_store = ProjectStore()
        self.faiss_store = FAISSStore()

        # Build tracer for phase-4.2 performance tracking
        self.build_tracer: BuildTracer = get_build_tracer()

        # Throttling control
        self._concurrent_build_limit = max_concurrent_rebuilds
        self._enqueued_builds: int = 0

        # Webhook dispatch (set externally)
        self.webhook_dispatcher: Optional[Callable] = None

    def start(self) -> None:
        """Start the rebuild coordinator worker thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="rebuild-coordinator")
        self._worker_thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the rebuild coordinator."""
        if not self._running:
            return

        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        self._running = False

    def queue_rebuild(
        self,
        project_id: str,
        trigger: RebuildTrigger = RebuildTrigger.MANUAL,
        changed_files: Optional[list[str]] = None,
        priority: int = 0,
        incremental: Optional[bool] = None,
    ) -> RebuildJob:
        """Queue a rebuild job for a project.

        Args:
            project_id: Project to rebuild
            trigger: What triggered this rebuild
            changed_files: List of changed file paths (relative to project)
            priority: Job priority (higher = more urgent)
            incremental: Whether to use incremental rebuild (overrides default)

        Returns:
            The queued RebuildJob
        """
        if incremental is None:
            incremental = self.enable_incremental

        job = RebuildJob(
            project_id=project_id,
            trigger=trigger,
            changed_files=changed_files or [],
            priority=priority,
            incremental=incremental,
        )

        with self._lock:
            # Check if there's already a pending/running job for this project
            existing = self._find_job_for_project(project_id)
            if existing and existing.status in (RebuildStatus.PENDING, RebuildStatus.QUEUED):
                # Update existing job with new info
                if changed_files:
                    existing.changed_files.extend(f for f in changed_files if f not in existing.changed_files)
                existing.priority = max(existing.priority, priority)
                existing.trigger = trigger
                return existing

            self._job_queue.append(job)
            # Sort by priority (highest first)
            self._job_queue = deque(sorted(self._job_queue, key=lambda j: -j.priority))
            job.status = RebuildStatus.QUEUED

        return job

    def _find_job_for_project(self, project_id: str) -> Optional[RebuildJob]:
        """Find existing job for a project."""
        for job in self._job_queue:
            if job.project_id == project_id:
                return job
        for job in self._running_jobs.values():
            if job.project_id == project_id:
                return job
        return None

    def _worker_loop(self) -> None:
        """Main worker loop processing rebuild jobs."""
        while not self._stop_event.is_set():
            job = self._get_next_job()
            if job is None:
                time.sleep(0.5)  # Wait for jobs
                continue

            self._execute_job(job)

    def _get_next_job(self) -> Optional[RebuildJob]:
        """Get next job from queue if capacity allows."""
        with self._lock:
            if len(self._running_jobs) >= self.max_concurrent_rebuilds:
                return None

            if not self._job_queue:
                return None

            job = self._job_queue.popleft()
            job.status = RebuildStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            self._running_jobs[job.job_id] = job
            return job

    def _execute_job(self, job: RebuildJob) -> None:
        """Execute a rebuild job with build tracing."""
        # Record build start
        build = self.build_tracer.start_build(
            build_id=f"rb-{job.job_id}",
            project_id=job.project_id,
            trigger=job.trigger.value,
            total_stages=3 if job.incremental else 4,
        )

        try:
            # Stage 1: Project extraction (if needed for incremental)
            if job.incremental and job.changed_files:
                stage = self.build_tracer.start_stage(build.build_id, "extract")
                try:
                    project_path = get_project(job.project_id)
                    # Load existing index for incremental diff
                    metadata, existing_chunks = self.project_store.load(project_path)
                except Exception as e:
                    self.build_tracer.complete_stage(
                        build.build_id,
                        "extract",
                        status="failed",
                        error=str(e)[:500],
                    )
                    raise
                self.build_tracer.complete_stage(build.build_id, "extract", status="completed")
            else:
                # Full rebuild - quick project lookup for stage 2
                project_path = get_project(job.project_id)

            # Stage 2: Parse and chunk
            stage = self.build_tracer.start_stage(build.build_id, "parse_chunk")
            try:
                if not job.incremental:
                    # Full rebuild - load existing to get file list
                    metadata, existing_chunks = self.project_store.load(project_path)
                else:
                    metadata, existing_chunks = self.project_store.load(project_path)
                    existing_chunk_map = {c.get("file_path", ""): c for c in existing_chunks}

                # Find all Python files for full rebuild
                if not job.incremental:
                    python_files = list(project_path.rglob("*.py"))
                    python_files = [f for f in python_files if ".trace" not in f.parts]
                    chunks: list[dict] = []
                    parse_errors: list[dict] = []

                    for python_file in python_files:
                        parsed = parse_python_file(python_file, project_path)
                        chunks.extend(build_code_chunks(parsed))
                        if parsed["syntax_error"]:
                            parse_errors.append({"file": parsed["file_path"], "error": parsed["syntax_error"]})
                else:
                    # Incremental - only rebuild changed files
                    chunks_to_remove = set()
                    chunks_to_rebuild = []

                    for changed_file in job.changed_files:
                        file_path = project_path / changed_file
                        if not file_path.exists():
                            for fp in list(existing_chunk_map.keys()):
                                if fp == changed_file or fp.startswith(changed_file + "/"):
                                    chunks_to_remove.add(fp)
                        else:
                            chunks_to_rebuild.append(changed_file)
                            for fp in list(existing_chunk_map.keys()):
                                if fp == changed_file or fp.startswith(changed_file + "/"):
                                    chunks_to_remove.add(fp)

                    # Remove old chunks
                    new_chunks = [c for c in existing_chunks if c.get("file_path", "") not in chunks_to_remove]

                    # Rebuild changed files
                    for changed_file in chunks_to_rebuild:
                        file_path = project_path / changed_file
                        if file_path.exists():
                            parsed = parse_python_file(file_path, project_path)
                            new_chunks.extend(build_code_chunks(parsed))

                    chunks = new_chunks

                self.build_tracer.complete_stage(build.build_id, "parse_chunk", status="completed")
                job.progress = 0.7

            except Exception as e:
                self.build_tracer.complete_stage(
                    build.build_id,
                    "parse_chunk",
                    status="failed",
                    error=str(e)[:500],
                )
                raise

            # Stage 3: Build semantic index
            stage = self.build_tracer.start_stage(build.build_id, "semantic_index")
            try:
                semantic_descriptor = {"status": "unavailable", "error": "Not attempted"}
                try:
                    settings = EmbeddingSettings.from_env()
                    semantic_descriptor = self._build_semantic_index(project_path, job.project_id, chunks, settings)
                except (EmbeddingError, Exception) as exc:
                    semantic_descriptor = {"status": "unavailable", "error": str(exc)[:500]}

                self.build_tracer.complete_stage(build.build_id, "semantic_index", status="completed")
                job.progress = 0.9

            except Exception as e:
                self.build_tracer.complete_stage(
                    build.build_id,
                    "semantic_index",
                    status="failed",
                    error=str(e)[:500],
                )
                raise

            # Stage 4: Save index
            stage = self.build_tracer.start_stage(build.build_id, "persist")
            try:
                metadata.update({
                    "project_id": job.project_id,
                    "file_count": sum(1 for p in project_path.rglob("*") if p.is_file() and ".trace" not in p.relative_to(project_path).parts),
                    "python_file_count": len(python_files) if not job.incremental else metadata.get("python_file_count", 0),
                    "indexed_chunk_count": len(chunks),
                    "parse_errors": parse_errors if not job.incremental else metadata.get("parse_errors", []),
                    "retrieval_backend": "faiss-cosine+lexical-rrf" if semantic_descriptor.get("status") == "ready" else "lexical-token-overlap",
                    "semantic_index": semantic_descriptor,
                })

                self.project_store.save(project_path, metadata, chunks)
                self.build_tracer.complete_stage(build.build_id, "persist", status="completed")
                job.progress = 1.0

            except Exception as e:
                self.build_tracer.complete_stage(
                    build.build_id,
                    "persist",
                    status="failed",
                    error=str(e)[:500],
                )
                raise

            # Build completed successfully
            build.status = "completed"
            job.status = RebuildStatus.COMPLETED
            job.progress = 1.0

        except Exception as e:
            # Build failed
            job.status = RebuildStatus.FAILED
            job.error = f"{type(e).__name__}: {str(e)[:500]}"
            build.status = "failed"
            build.error = job.error
            build.exceptions_raised += 1

        finally:
            job.completed_at = datetime.now(timezone.utc)
            job.progress = 1.0

            build.completed_at = datetime.now(timezone.utc)

            with self._lock:
                self._running_jobs.pop(job.job_id, None)
                self._completed_jobs.append(job)

            # Fire callback
            if self.rebuild_callback:
                try:
                    self.rebuild_callback(job)
                except Exception:
                    pass

            # Fire webhook
            if self.webhook_dispatcher:
                try:
                    self.webhook_dispatcher(job)
                except Exception:
                    pass

            # Finalize build record
            self.build_tracer.complete_build(build.build_id, build.status, build.error)

    def _rebuild_project(self, job: RebuildJob) -> None:
        """Perform the actual project rebuild."""
        project_id = job.project_id
        project_path = get_project(project_id)

        if project_path is None:
            raise ValueError(f"Project not found: {project_id}")

        # Update progress
        job.progress = 0.1

        # Extract changed files
        changed_files = job.changed_files

        if job.incremental and changed_files:
            # Incremental rebuild
            self._incremental_rebuild(job, project_path, changed_files)
        else:
            # Full rebuild
            self._full_rebuild(job, project_path)

    def _full_rebuild(self, job: RebuildJob, project_path: Path) -> None:
        """Full project rebuild - re-extract, re-parse, re-chunk, re-embed."""
        project_id = job.project_id

        job.progress = 0.2

        # Load existing metadata and chunks to get file list
        try:
            metadata, existing_chunks = self.project_store.load(project_path)
        except (ProjectIndexMissing, ProjectIndexError):
            metadata = {}
            existing_chunks = []

        job.progress = 0.3

        # Find all Python files in project
        python_files = list(project_path.rglob("*.py"))
        python_files = [f for f in python_files if ".trace" not in f.parts]

        job.progress = 0.4

        # Parse and chunk all files
        chunks: list[dict] = []
        parse_errors: list[dict] = []

        for python_file in python_files:
            parsed = parse_python_file(python_file, project_path)
            chunks.extend(build_code_chunks(parsed))
            if parsed["syntax_error"]:
                parse_errors.append({"file": parsed["file_path"], "error": parsed["syntax_error"]})

        job.progress = 0.6

        # Build semantic index if enabled
        semantic_descriptor = {"status": "unavailable", "error": "Not attempted"}
        try:
            settings = EmbeddingSettings.from_env()
            semantic_descriptor = self._build_semantic_index(project_path, project_id, chunks, settings)
        except (EmbeddingError, Exception) as exc:
            semantic_descriptor = {"status": "unavailable", "error": str(exc)[:500]}

        job.progress = 0.8

        # Update metadata
        metadata.update({
            "project_id": project_id,
            "file_count": sum(1 for p in project_path.rglob("*") if p.is_file() and ".trace" not in p.relative_to(project_path).parts),
            "python_file_count": len(python_files),
            "indexed_chunk_count": len(chunks),
            "parse_errors": parse_errors,
            "retrieval_backend": "faiss-cosine+lexical-rrf" if semantic_descriptor.get("status") == "ready" else "lexical-token-overlap",
            "semantic_index": semantic_descriptor,
        })

        # Save atomically
        self.project_store.save(project_path, metadata, chunks)
        job.progress = 1.0

    def _incremental_rebuild(self, job: RebuildJob, project_path: Path, changed_files: list[str]) -> None:
        """Incremental rebuild - only reparse changed files."""
        project_id = job.project_id

        job.progress = 0.2

        # Load existing index
        metadata, existing_chunks = self.project_store.load(project_path)
        existing_chunk_map = {c.get("file_path", ""): c for c in existing_chunks}

        job.progress = 0.3

        # Determine which chunks need rebuilding
        chunks_to_remove = set()
        chunks_to_rebuild = []

        for changed_file in changed_files:
            file_path = project_path / changed_file
            if not file_path.exists():
                # File deleted - remove its chunks
                for fp in list(existing_chunk_map.keys()):
                    if fp == changed_file or fp.startswith(changed_file + "/"):
                        chunks_to_remove.add(fp)
            else:
                # File changed - reparse
                chunks_to_rebuild.append(changed_file)
                for fp in list(existing_chunk_map.keys()):
                    if fp == changed_file or fp.startswith(changed_file + "/"):
                        chunks_to_remove.add(fp)

        job.progress = 0.4

        # Remove old chunks
        new_chunks = [c for c in existing_chunks if c.get("file_path", "") not in chunks_to_remove]

        # Rebuild changed files
        for changed_file in chunks_to_rebuild:
            file_path = project_path / changed_file
            if file_path.exists():
                parsed = parse_python_file(file_path, project_path)
                new_chunks.extend(build_code_chunks(parsed))
                if parsed["syntax_error"]:
                    # Update parse errors
                    pass

        job.progress = 0.6

        # Rebuild semantic index for all chunks
        try:
            settings = EmbeddingSettings.from_env()
            semantic_descriptor = self._build_semantic_index(project_path, project_id, new_chunks, settings)
        except (EmbeddingError, Exception) as exc:
            semantic_descriptor = {"status": "unavailable", "error": str(exc)[:500]}

        job.progress = 0.8

        # Update metadata
        metadata.update({
            "indexed_chunk_count": len(new_chunks),
            "retrieval_backend": "faiss-cosine+lexical-rrf" if semantic_descriptor.get("status") == "ready" else "lexical-token-overlap",
            "semantic_index": semantic_descriptor,
        })

        # Save atomically
        self.project_store.save(project_path, metadata, new_chunks)
        job.progress = 1.0

    def _build_semantic_index(
        self,
        project_path: Path,
        project_id: str,
        chunks: list[dict],
        settings: EmbeddingSettings,
    ) -> dict:
        """Build semantic index for project."""
        import numpy as np

        ordered = canonical_chunks(chunks)
        embedder = SentenceTransformerEmbedder(settings)

        if ordered:
            vectors, windows = embedder.encode_chunk_documents(ordered)
        else:
            vectors = np.empty((0, 0), dtype=np.float32)
            windows = []

        embedding_config = embedder.index_config
        header = self.faiss_store.save(
            project_path,
            project_id,
            ordered,
            vectors,
            model_name=settings.model_name,
            embedding_config=embedding_config,
            vector_mapping=windows,
        )

        return {
            "status": "ready",
            "model_name": settings.model_name,
            "embedding_config": embedding_config,
            "dimension": header["dimension"],
            "chunk_count": len(ordered),
            "window_count": header["count"],
            "corpus_fingerprint": header["corpus_fingerprint"],
            "index_file": FAISSStore.INDEX_NAME,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_status(self) -> dict:
        """Get coordinator status."""
        with self._lock:
            return {
                "running": self._running,
                "max_concurrent": self.max_concurrent_rebuilds,
                "enable_incremental": self.enable_incremental,
                "queue_length": len(self._job_queue),
                "running_jobs": len(self._running_jobs),
                "completed_jobs": len(self._completed_jobs),
            }

    def get_jobs(self, status: Optional[RebuildStatus] = None, limit: int = 50) -> list[dict]:
        """Get recent jobs, optionally filtered by status."""
        with self._lock:
            all_jobs = list(self._job_queue) + list(self._running_jobs.values()) + list(self._completed_jobs)
            if status:
                all_jobs = [j for j in all_jobs if j.status == status]
            all_jobs.sort(key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in all_jobs[:limit]]

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get a specific job by ID."""
        with self._lock:
            for job in self._job_queue:
                if job.job_id == job_id:
                    return job.to_dict()
            for job in self._running_jobs.values():
                if job.job_id == job_id:
                    return job.to_dict()
            for job in self._completed_jobs:
                if job.job_id == job_id:
                    return job.to_dict()
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        with self._lock:
            for job in self._job_queue:
                if job.job_id == job_id:
                    job.status = RebuildStatus.CANCELLED
                    job.completed_at = datetime.now(timezone.utc)
                    self._job_queue.remove(job)
                    self._completed_jobs.append(job)
                    return True
        return False

    def cancel_jobs_for_project(self, project_id: str) -> int:
        """Cancel all queued rebuild jobs for a project."""
        cancelled = 0
        with self._lock:
            to_remove = [j for j in self._job_queue if j.project_id == project_id]
            for job in to_remove:
                job.status = RebuildStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                self._job_queue.remove(job)
                self._completed_jobs.append(job)
                cancelled += 1
        return cancelled

    def clear_history(self) -> int:
        """Clear completed job history. Returns count cleared."""
        with self._lock:
            count = len(self._completed_jobs)
            self._completed_jobs.clear()
            return count


# Global coordinator instance
_rebuild_coordinator: Optional[RebuildCoordinator] = None


def get_rebuild_coordinator() -> Optional[RebuildCoordinator]:
    """Get the global rebuild coordinator."""
    return _rebuild_coordinator


def set_rebuild_coordinator(coordinator: RebuildCoordinator) -> None:
    """Set the global rebuild coordinator."""
    global _rebuild_coordinator
    _rebuild_coordinator = coordinator