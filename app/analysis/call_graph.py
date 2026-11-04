"""Approximate same-project Python call-site search."""

from __future__ import annotations

import ast
from pathlib import Path


def _callee_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _CallCollector(ast.NodeVisitor):
    def __init__(self, target: str, file_path: str, source: str = ""):
        self.target = target
        self.file_path = file_path
        self.source = source
        self.calls: list[dict] = []
        self.scope: list[str] = []

    def _visit_function(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_function(node, "<lambda>")

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee_name(node.func)
        if name == self.target:
            self.calls.append({
                "file": self.file_path,
                "line": node.lineno,
                "caller": ".".join(self.scope) or "<module>",
                "expression": ast.get_source_segment(self.source, node) or name,
            })
        self.generic_visit(node)


def find_function_calls(project_path: str | Path, function_name: str) -> list[dict]:
    root = Path(project_path).resolve()
    target = function_name.rsplit(".", 1)[-1]
    results: list[dict] = []
    if not root.is_dir():
        return results
    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=path.relative_to(root).as_posix())
        except (OSError, SyntaxError, ValueError):
            continue
        visitor = _CallCollector(target, path.relative_to(root).as_posix(), source=source)
        visitor.visit(tree)
        results.extend(visitor.calls)
    return sorted(results, key=lambda item: (item["file"], item["line"]))
