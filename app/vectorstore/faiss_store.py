"""Atomic per-project exact-search FAISS index with a validated manifest."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SemanticIndexError(ValueError):
    """A project's semantic index cannot safely be used."""


class SemanticIndexMissing(SemanticIndexError):
    """The project does not have a semantic index yet."""


class SemanticIndexStale(SemanticIndexError):
    """The semantic index was built from another model or corpus."""


_MAGIC = b"TRACE-FAISS-INDEX\x00\x01"
_FORMAT = "trace-faiss-flat-ip-v2"
_MAPPING_FIELDS = (
    "chunk_id", "file_path", "kind", "symbol", "line_start", "line_end", "part",
)
_WINDOW_MAPPING_FIELDS = (
    "chunk_id", "parent_chunk_id", "file_path", "kind", "symbol", "line_start", "line_end",
    "part", "window_id", "window_index", "window_count", "char_start", "char_end",
    "token_start", "token_end",
)


def canonical_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable, ID-sorted corpus and reject ambiguous chunk IDs."""
    by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            raise SemanticIndexError("Every semantic chunk must have a chunk_id")
        current = dict(chunk)
        previous = by_id.get(chunk_id)
        if previous is not None and previous != current:
            raise SemanticIndexError(f"Chunk ID {chunk_id!r} refers to conflicting source")
        by_id[chunk_id] = current
    return [by_id[chunk_id] for chunk_id in sorted(by_id)]


def corpus_fingerprint(chunks: list[dict[str, Any]]) -> str:
    canonical = canonical_chunks(chunks)
    try:
        payload = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticIndexError("Chunk corpus is not valid JSON metadata") from exc
    return hashlib.sha256(payload).hexdigest()


def _mapping_for(
    chunks: list[dict[str, Any]],
    vector_mapping: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ordered = canonical_chunks(chunks)
    if vector_mapping is None:
        return [{field: chunk.get(field) for field in _MAPPING_FIELDS} for chunk in ordered]

    parents = {str(chunk["chunk_id"]) for chunk in ordered}
    seen_windows: set[str] = set()
    mapping: list[dict[str, Any]] = []
    for window in vector_mapping:
        parent_id = str(window.get("parent_chunk_id") or window.get("chunk_id") or "")
        if not parent_id or parent_id not in parents:
            raise SemanticIndexError("Every embedding window must refer to a source chunk in this project")
        window_id = str(window.get("window_id") or "")
        if not window_id or window_id in seen_windows:
            raise SemanticIndexError("Every embedding window must have a unique window_id")
        seen_windows.add(window_id)
        current = {field: window.get(field) for field in _WINDOW_MAPPING_FIELDS}
        current["chunk_id"] = parent_id
        current["parent_chunk_id"] = parent_id
        for field in ("char_start", "char_end", "token_start", "token_end", "line_start", "line_end"):
            if not isinstance(current.get(field), int) or current[field] < 0:
                raise SemanticIndexError(f"Embedding window {field} must be a non-negative integer")
        if current["char_end"] <= current["char_start"] or current["token_end"] <= current["token_start"]:
            raise SemanticIndexError("Embedding windows must contain at least one source token and character")
        if current["line_end"] < current["line_start"]:
            raise SemanticIndexError("Embedding window source line range is invalid")
        mapping.append(current)
    return mapping


@dataclass
class LoadedSemanticIndex:
    """Validated project-local FAISS index and its stable row mapping."""

    faiss_index: Any | None
    mapping: list[dict[str, Any]]
    header: dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.mapping)

    @property
    def dimension(self) -> int:
        return int(self.header["dimension"])

    def search(self, query_vector: Any, top_k: int) -> list[dict[str, Any]]:
        import numpy as np

        if top_k <= 0 or self.count == 0 or self.faiss_index is None:
            return []
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, self.dimension):
            raise SemanticIndexError("Query embedding dimension does not match the index")
        if not np.isfinite(vector).all():
            raise SemanticIndexError("Query embedding contains non-finite values")
        norm = float(np.linalg.norm(vector[0]))
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise SemanticIndexError("Query embedding must be L2-normalized")

        scores, positions = self.faiss_index.search(
            np.ascontiguousarray(vector, dtype=np.float32),
            min(int(top_k), self.count),
        )
        hits = []
        for score, position in zip(scores[0].tolist(), positions[0].tolist()):
            if position < 0 or position >= self.count:
                continue
            hits.append({**self.mapping[position], "score": float(score)})
        hits.sort(key=lambda item: (-item["score"], str(item["chunk_id"])))
        return hits


class FAISSStore:
    """Persist an exact IndexFlatIP whose unit vectors produce cosine scores."""

    INDEX_NAME = "semantic.faiss"

    @staticmethod
    def index_path(project_path: str | Path) -> Path:
        return Path(project_path).resolve() / ".trace" / FAISSStore.INDEX_NAME

    def save(
        self,
        project_path: str | Path,
        project_id: str,
        chunks: list[dict[str, Any]],
        vectors: Any,
        *,
        model_name: str,
        embedding_config: dict[str, Any],
        vector_mapping: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import faiss
        import numpy as np

        project = Path(project_path).resolve()
        if project.name != project_id:
            raise SemanticIndexError("Project ID does not match the index directory")
        ordered = canonical_chunks(chunks)
        mapping = _mapping_for(ordered, vector_mapping)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(mapping):
            raise SemanticIndexError("Embedding rows do not match the semantic window mapping")
        if not np.isfinite(matrix).all():
            raise SemanticIndexError("Embedding matrix contains non-finite values")

        if mapping:
            if matrix.shape[1] <= 0:
                raise SemanticIndexError("Embedding dimension must be positive")
            norms = np.linalg.norm(matrix, axis=1)
            if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-3):
                raise SemanticIndexError("Document vectors must be L2-normalized")
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            index = faiss.IndexFlatIP(int(matrix.shape[1]))
            index.add(matrix)
            serialized = faiss.serialize_index(index).tobytes()
            dimension = int(matrix.shape[1])
        else:
            if matrix.shape[1] not in (0,):
                raise SemanticIndexError("An index without embedding windows must have dimension zero")
            serialized = b""
            dimension = 0

        header = {
            "format": _FORMAT,
            "project_id": project_id,
            "model_name": model_name,
            "embedding_config": embedding_config,
            "dimension": dimension,
            "metric": "inner_product_on_l2_normalized_vectors",
            "normalization": "l2-unit",
            "corpus_fingerprint": corpus_fingerprint(ordered),
            "count": len(mapping),
            "chunk_count": len(ordered),
            "mapping": mapping,
        }
        header_bytes = json.dumps(
            header,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(header_bytes) > 64 * 1024 * 1024:
            raise SemanticIndexError("Semantic index manifest is too large")

        trace_dir = project / ".trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        target = trace_dir / self.INDEX_NAME
        handle, temporary_name = tempfile.mkstemp(prefix="semantic-", suffix=".tmp", dir=trace_dir)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(_MAGIC)
                stream.write(struct.pack(">Q", len(header_bytes)))
                stream.write(header_bytes)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return header

    def load(
        self,
        project_path: str | Path,
        project_id: str,
        chunks: list[dict[str, Any]],
        *,
        model_name: str,
        embedding_config: dict[str, Any],
        vector_mapping: list[dict[str, Any]] | None = None,
    ) -> LoadedSemanticIndex:
        path = self.index_path(project_path)
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise SemanticIndexMissing("Project semantic index is missing") from exc
        except OSError as exc:
            raise SemanticIndexError("Project semantic index cannot be read") from exc

        try:
            fixed_size = len(_MAGIC) + 8
            if len(payload) < fixed_size or payload[:len(_MAGIC)] != _MAGIC:
                raise SemanticIndexError("Semantic index has an invalid file header")
            header_size = struct.unpack(">Q", payload[len(_MAGIC):fixed_size])[0]
            end = fixed_size + header_size
            if header_size > 64 * 1024 * 1024 or end > len(payload):
                raise SemanticIndexError("Semantic index manifest is truncated or too large")
            header = json.loads(payload[fixed_size:end].decode("utf-8"))
            index_bytes = payload[end:]
            if not isinstance(header, dict) or header.get("format") != _FORMAT:
                raise SemanticIndexError("Semantic index manifest has an unsupported format")
            if Path(project_path).resolve().name != project_id or header.get("project_id") != project_id:
                raise SemanticIndexStale("Semantic index belongs to a different project")
            if header.get("model_name") != model_name:
                raise SemanticIndexStale("Semantic index was built with a different model")
            if header.get("embedding_config") != embedding_config:
                raise SemanticIndexStale("Semantic index embedding configuration has changed")
            expected_fingerprint = corpus_fingerprint(chunks)
            if header.get("corpus_fingerprint") != expected_fingerprint:
                raise SemanticIndexStale("Semantic index does not match the current chunk corpus")

            ordered = canonical_chunks(chunks)
            expected_mapping = _mapping_for(ordered, vector_mapping)
            if (
                header.get("mapping") != expected_mapping
                or header.get("count") != len(expected_mapping)
                or header.get("chunk_count") != len(ordered)
            ):
                raise SemanticIndexStale("Semantic index source mapping does not match the project")
            dimension = header.get("dimension")
            if not isinstance(dimension, int) or dimension < 0:
                raise SemanticIndexError("Semantic index dimension is invalid")
            if header.get("metric") != "inner_product_on_l2_normalized_vectors":
                raise SemanticIndexError("Semantic index similarity metric is unsupported")
            if header.get("normalization") != "l2-unit":
                raise SemanticIndexError("Semantic index normalization is unsupported")

            if not expected_mapping:
                if dimension != 0 or index_bytes:
                    raise SemanticIndexError("Empty semantic index has unexpected vector data")
                return LoadedSemanticIndex(None, expected_mapping, header)
            if dimension <= 0 or not index_bytes:
                raise SemanticIndexError("Semantic index vector data is missing")

            import faiss
            import numpy as np

            index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8).copy())
            if not isinstance(index, faiss.IndexFlatIP):
                raise SemanticIndexError("FAISS index is not an exact IndexFlatIP index")
            if index.d != dimension or index.ntotal != len(expected_mapping):
                raise SemanticIndexError("FAISS index shape does not match its manifest")
            if index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise SemanticIndexError("FAISS index does not use inner-product search")
            return LoadedSemanticIndex(index, expected_mapping, header)
        except SemanticIndexError:
            raise
        except Exception as exc:
            raise SemanticIndexError(f"Semantic index is corrupt or incompatible: {exc}") from exc
