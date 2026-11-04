"""Python AST parser that retains source locations and tolerates syntax errors."""

from __future__ import annotations

import ast
from pathlib import Path


class _DefinitionCollector(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.definitions: list[dict] = []
        self.stack: list[str] = []

    def _visit_definition(self, node: ast.AST, kind: str, name: str) -> None:
        qualified_name = ".".join([*self.stack, name])
        code = ast.get_source_segment(self.source, node) or ""
        self.definitions.append({
            "kind": kind,
            "name": name,
            "qualified_name": qualified_name,
            "parent": ".".join(self.stack) or None,
            "line_start": int(getattr(node, "lineno", 1)),
            "line_end": int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
            "code": code,
        })
        self.stack.append(name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node, "function", node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition(node, "function", node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_definition(node, "class", node.name)


def _relative_path(path: Path, project_root: str | Path | None) -> str:
    if project_root is not None:
        try:
            return path.resolve().relative_to(Path(project_root).resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def parse_python_file(path: str | Path, project_root: str | Path | None = None) -> dict:
    source_path = Path(path)
    source = source_path.read_text(encoding="utf-8", errors="replace")
    result = {
        "file_path": _relative_path(source_path, project_root),
        "source": source,
        "definitions": [],
        "imports": [],
        "syntax_error": None,
    }
    try:
        tree = ast.parse(source, filename=result["file_path"])
    except (SyntaxError, ValueError) as exc:
        result["syntax_error"] = str(exc)
        return result

    collector = _DefinitionCollector(source)
    collector.visit(tree)
    result["definitions"] = collector.definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result["imports"].extend({"module": alias.name, "line": node.lineno} for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            result["imports"].extend({"module": module, "name": alias.name, "line": node.lineno} for alias in node.names)
    result["_top_level_definition_lines"] = [
        (int(node.lineno), int(getattr(node, "end_lineno", node.lineno)))
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return result
