"""Locate Python definitions by name, including nested qualified names."""

from __future__ import annotations

from pathlib import Path

from app.parsing.ast_parser import parse_python_file


def get_function_code(project_path: str | Path, function_name: str) -> dict | None:
    root = Path(project_path).resolve()
    matches: list[tuple[int, str, dict]] = []
    requested = function_name.strip()
    for path in sorted(root.rglob("*.py")) if root.is_dir() else []:
        parsed = parse_python_file(path, root)
        for definition in parsed["definitions"]:
            if definition["kind"] != "function":
                continue
            if definition["name"] == requested or definition["qualified_name"] == requested:
                matches.append((int(definition["line_start"]), parsed["file_path"], definition))
    if not matches:
        return None
    line, relative, definition = matches[0]
    return {
        "name": definition["name"],
        "qualified_name": definition["qualified_name"],
        "file": relative,
        "line": line,
        "code": definition["code"],
        "ambiguous": len(matches) > 1,
        "matches": [{"file": item[1], "line": item[0], "qualified_name": item[2]["qualified_name"]} for item in matches],
    }
