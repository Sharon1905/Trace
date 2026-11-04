"""Atomic, project-local persistence for metadata and indexed source chunks."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ProjectIndexError(ValueError):
    """Raised when a project's persisted index is missing or invalid."""


class ProjectIndexMissing(ProjectIndexError):
    """Raised when a project has no persisted index yet."""


class ProjectStore:
    """Read and write the index contained in one project's ``.trace`` folder."""

    INDEX_NAME = "index.json.gz"

    def save(self, project_path: str | Path, metadata: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        project = Path(project_path).resolve()
        trace_dir = project / ".trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        target = trace_dir / self.INDEX_NAME
        handle, temporary_name = tempfile.mkstemp(prefix="index-", suffix=".tmp", dir=trace_dir)
        temporary = Path(temporary_name)
        try:
            payload = json.dumps(
                {"metadata": metadata, "chunks": chunks},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            with os.fdopen(handle, "wb") as stream:
                with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                    compressed.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, project_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        project = Path(project_path).resolve()
        index_path = project / ".trace" / self.INDEX_NAME
        try:
            with gzip.open(index_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError as exc:
            raise ProjectIndexMissing("Project has not been indexed") from exc
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProjectIndexError("Project index is corrupt or unreadable") from exc

        if not isinstance(payload, dict):
            raise ProjectIndexError("Project index has an invalid format")
        metadata = payload.get("metadata")
        chunks = payload.get("chunks")
        if not isinstance(metadata, dict) or not isinstance(chunks, list):
            raise ProjectIndexError("Project index has an invalid format")
        if metadata.get("project_id") != project.name:
            raise ProjectIndexError("Project index does not match its project directory")
        if any(not isinstance(chunk, dict) for chunk in chunks):
            raise ProjectIndexError("Project index contains invalid chunks")
        return metadata, chunks
