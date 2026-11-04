"""Project paths and storage root helpers; selection is always explicit."""

from __future__ import annotations

import os
import re
from pathlib import Path

_PROJECT_ID = re.compile(r"^[0-9a-f]{32}$")


def get_data_dir() -> Path:
    """Return TRACE's data root (default: ``./data``)."""
    return Path(os.environ.get("TRACE_DATA_DIR", "data")).expanduser().resolve()


def get_project(project_id: str, data_root: str | Path | None = None) -> Path | None:
    if not _PROJECT_ID.fullmatch(project_id):
        return None
    root = Path(data_root).resolve() if data_root is not None else get_data_dir()
    projects = (root / "projects").resolve()
    path = (projects / project_id).resolve()
    if path.parent != projects or not path.is_dir():
        return None
    return path


def list_projects(data_root: str | Path | None = None) -> list[str]:
    root = Path(data_root).resolve() if data_root is not None else get_data_dir()
    projects = (root / "projects").resolve()
    if not projects.is_dir():
        return []
    result = []
    for entry in projects.iterdir():
        if entry.is_dir() and _PROJECT_ID.fullmatch(entry.name):
            if (entry / ".trace" / "index.json.gz").is_file():
                result.append(entry.name)
    return result
