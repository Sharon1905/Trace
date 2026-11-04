"""Safe, bounded extraction of ZIP projects."""

from __future__ import annotations

import os
import shutil
import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from app.utils.project_manager import get_data_dir

def _configured_limit(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


MAX_ARCHIVE_BYTES = _configured_limit("TRACE_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
MAX_EXPANDED_BYTES = _configured_limit("TRACE_MAX_EXTRACTED_BYTES", 100 * 1024 * 1024)
MAX_MEMBERS = _configured_limit("TRACE_MAX_ARCHIVE_ENTRIES", 10_000)
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class ExtractionError(ValueError):
    """Raised when an uploaded archive is invalid or exceeds safety limits."""


def extract_project(archive_path: str | Path, data_root: str | Path | None = None) -> tuple[str, list[Path]]:
    """Extract a project into an isolated directory and return its Python files.

    All archive paths are validated before writing. Unsupported files are kept so
    repository explanations can describe the project, but only ``.py`` files are
    returned for indexing.
    """
    archive = Path(archive_path)
    if not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ExtractionError("Archive is missing or exceeds the 25 MiB upload limit")
    if not zipfile.is_zipfile(archive):
        raise ExtractionError("Uploaded file is not a valid ZIP archive")

    root = Path(data_root) if data_root is not None else get_data_dir()
    projects = root / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    project_id = uuid.uuid4().hex
    destination = projects / project_id
    destination.mkdir()

    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            if len(members) > MAX_MEMBERS:
                raise ExtractionError("Archive contains too many entries")
            total_size = sum(item.file_size for item in members)
            if total_size > MAX_EXPANDED_BYTES:
                raise ExtractionError("Archive expands beyond the 100 MiB limit")

            checked: list[tuple[zipfile.ZipInfo, Path]] = []
            seen_paths: set[str] = set()
            for item in members:
                normalized = item.filename.replace("\\", "/")
                relative = PurePosixPath(normalized)
                if not relative.parts or relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
                    raise ExtractionError(f"Unsafe archive path: {item.filename}")
                if relative.parts[0].casefold() == ".trace":
                    raise ExtractionError("Archive path uses a reserved TRACE metadata directory")
                if any(
                    ":" in part
                    or part.endswith((".", " "))
                    or part.rstrip(" .").split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
                    for part in relative.parts
                ):
                    raise ExtractionError(f"Unsafe archive path: {item.filename}")
                collision_key = "/".join(relative.parts).casefold()
                if collision_key in seen_paths:
                    raise ExtractionError(f"Archive contains a duplicate path: {item.filename}")
                seen_paths.add(collision_key)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ExtractionError(f"Symbolic links are not allowed: {item.filename}")
                target = destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ExtractionError(f"Unsafe archive path: {item.filename}")
                checked.append((item, target))

            for item, target in checked:
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

        python_files = sorted(
            (path for path in destination.rglob("*") if path.is_file() and path.suffix.casefold() == ".py"),
            key=lambda path: path.relative_to(destination).as_posix().casefold(),
        )
        return project_id, python_files
    except (zipfile.BadZipFile, EOFError, RuntimeError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise ExtractionError("Uploaded file is not a valid, readable ZIP archive") from exc
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
