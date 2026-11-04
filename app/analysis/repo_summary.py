"""Deterministic repository summary."""

from __future__ import annotations

from pathlib import Path

from app.parsing.ast_parser import parse_python_file


def summarize_repository(project_path: str | Path) -> str:
    root = Path(project_path).resolve()
    if not root.is_dir():
        return "Repository directory is unavailable."
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and ".trace" not in p.relative_to(root).parts),
        key=lambda p: p.relative_to(root).as_posix().casefold(),
    )
    python_files = [path for path in files if path.suffix.casefold() == ".py"]
    rows = [f"Repository contains {len(files)} files, including {len(python_files)} Python files."]
    for path in python_files:
        parsed = parse_python_file(path, root)
        names = [definition["qualified_name"] for definition in parsed["definitions"]]
        note = ", ".join(names[:8]) if names else ("syntax error" if parsed["syntax_error"] else "no definitions")
        rows.append(f"- {parsed['file_path']}: {note}")
    other = [path.relative_to(root).as_posix() for path in files if path.suffix.casefold() != ".py"]
    if other:
        rows.append("Other files: " + ", ".join(other[:30]))
    return "\n".join(rows)
