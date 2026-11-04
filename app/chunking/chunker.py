"""Build source chunks aligned to Python definitions and module code."""

from __future__ import annotations

import hashlib


def _split_source(code: str, line_start: int, max_chars: int) -> list[tuple[int, int, str]]:
    lines = code.splitlines(keepends=True)
    if not lines and code:
        lines = [code]
    parts: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_len = 0
    start = line_start
    for offset, line in enumerate(lines):
        if current and current_len + len(line) > max_chars:
            parts.append((start, start + len(current) - 1, "".join(current)))
            current = []
            current_len = 0
            start = line_start + offset
        if len(line) > max_chars:
            if current:
                parts.append((start, start + len(current) - 1, "".join(current)))
                current = []
                current_len = 0
            for index in range(0, len(line), max_chars):
                parts.append((line_start + offset, line_start + offset, line[index:index + max_chars]))
            start = line_start + offset + 1
            continue
        if not current:
            start = line_start + offset
        current.append(line)
        current_len += len(line)
    if current:
        parts.append((start, start + len(current) - 1, "".join(current)))
    return parts


def build_code_chunks(parsed: dict, max_chars: int = 4000) -> list[dict]:
    if max_chars < 128:
        raise ValueError("max_chars must be at least 128")
    file_path = str(parsed["file_path"])
    source = str(parsed.get("source", ""))
    chunks: list[dict] = []

    def append_chunks(kind: str, symbol: str | None, code: str, line_start: int) -> None:
        for part_index, (start, end, part) in enumerate(_split_source(code, line_start, max_chars), 1):
            normalized = f"{file_path}\0{symbol or ''}\0{start}\0{part}"
            chunk_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
            chunks.append({
                "chunk_id": chunk_id,
                "file_path": file_path,
                "kind": kind,
                "symbol": symbol,
                "line_start": start,
                "line_end": end,
                "part": part_index,
                "code": part,
            })

    if parsed.get("syntax_error"):
        if source:
            append_chunks("unparsed", None, source, 1)
        return chunks

    for definition in parsed.get("definitions", []):
        code = str(definition.get("code", ""))
        if code.strip():
            append_chunks(
                str(definition["kind"]),
                str(definition["qualified_name"]),
                code,
                int(definition["line_start"]),
            )

    lines = source.splitlines(keepends=True)
    excluded: set[int] = set()
    for first, last in parsed.get("_top_level_definition_lines", []):
        excluded.update(range(int(first), int(last) + 1))
    module_rows = [line if number not in excluded else "" for number, line in enumerate(lines, 1)]
    module_code = "".join(module_rows)
    if module_code.strip():
        first_line = next((i for i, line in enumerate(module_rows, 1) if line.strip()), 1)
        append_chunks("module", None, module_code, first_line)
    return chunks
