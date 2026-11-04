"""Project-confined file lookup."""

from __future__ import annotations

from pathlib import Path


def get_file_content(project_path: str | Path, filename: str) -> tuple[str | None, str | None]:
    root = Path(project_path).resolve()
    if filename.replace("\\", "/").split("/", 1)[0].casefold() == ".trace":
        return None, None
    candidate = (root / filename.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None, None
    try:
        return candidate.relative_to(root).as_posix(), candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
