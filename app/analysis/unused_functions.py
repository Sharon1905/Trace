"""Find conservative candidates with no statically visible call sites."""

from __future__ import annotations

import ast
from pathlib import Path

from app.analysis.call_graph import _callee_name


def find_unused_functions(project_path: str | Path) -> list[dict]:
    root = Path(project_path).resolve()
    definitions: list[dict] = []
    called_names: set[str] = set()
    if not root.is_dir():
        return []
    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=path.relative_to(root).as_posix())
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _callee_name(node.func)
                if name:
                    called_names.add(name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                called_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                called_names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue
                definitions.append({
                    "function": node.name,
                    "qualified_name": node.name,
                    "file": path.relative_to(root).as_posix(),
                    "line": node.lineno,
                    "candidate": node.name not in called_names,
                })
    # Recompute after collecting all files so definitions earlier in sort order
    # are evaluated against calls found later in the project.
    for item in definitions:
        item["candidate"] = item["function"] not in called_names
    return [
        {**item, "note": "Static candidate only; reflection, framework hooks, and dynamic calls may use it."}
        for item in definitions
        if item["candidate"]
    ]
