"""Lazy Sentence Transformers adapter with safe token-aware code windows."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Protocol


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOWING_STRATEGY = "token-offset-overlap-v1"


class EmbeddingError(RuntimeError):
    """Embedding configuration, loading, tokenization, or output failed."""


class EmbeddingModel(Protocol):
    def encode_documents(self, texts: list[str]): ...

    def encode_queries(self, texts: list[str]): ...


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = "cpu"
    batch_size: int = 32
    window_overlap_tokens: int = 32

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        model_name = os.getenv("TRACE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        device = os.getenv("TRACE_EMBEDDING_DEVICE", "cpu").strip()
        try:
            batch_size = int(os.getenv("TRACE_EMBEDDING_BATCH_SIZE", "32"))
        except ValueError as exc:
            raise EmbeddingError("TRACE_EMBEDDING_BATCH_SIZE must be an integer") from exc
        try:
            overlap = int(os.getenv("TRACE_EMBEDDING_WINDOW_OVERLAP_TOKENS", "32"))
        except ValueError as exc:
            raise EmbeddingError("TRACE_EMBEDDING_WINDOW_OVERLAP_TOKENS must be an integer") from exc
        if not model_name:
            raise EmbeddingError("TRACE_EMBEDDING_MODEL must not be blank")
        if not device:
            raise EmbeddingError("TRACE_EMBEDDING_DEVICE must not be blank")
        if batch_size <= 0:
            raise EmbeddingError("TRACE_EMBEDDING_BATCH_SIZE must be greater than zero")
        if overlap < 0:
            raise EmbeddingError("TRACE_EMBEDDING_WINDOW_OVERLAP_TOKENS must not be negative")
        return cls(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            window_overlap_tokens=overlap,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "normalization": "l2-unit",
            "similarity": "cosine-via-inner-product",
            "document_encoding": "encode_document",
            "query_encoding": "encode_query",
            "windowing": {
                "strategy": WINDOWING_STRATEGY,
                "overlap_tokens": self.window_overlap_tokens,
            },
        }


def source_window_evidence(chunk: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    """Expand a token span to complete original source lines for readable citations."""
    source = str(chunk.get("code", ""))
    start = max(0, min(int(window.get("char_start", 0)), len(source)))
    end = max(start, min(int(window.get("char_end", len(source))), len(source)))
    first = source.rfind("\n", 0, start) + 1
    next_newline = source.find("\n", end)
    last = len(source) if next_newline < 0 else next_newline + 1
    evidence = source[first:last]
    base_line = int(chunk.get("line_start", 1))
    line_start = base_line + source.count("\n", 0, first)
    line_count = max(1, len(evidence.splitlines()))
    return {
        "code": evidence,
        "line_start": line_start,
        "line_end": line_start + line_count - 1,
    }


class SentenceTransformerEmbedder:
    """Load each model/device pair once and create normalized float32 vectors.

    Long source chunks are split into overlapping tokenizer windows. Each window
    embeds a repeated file/type/symbol prefix plus a slice of the original code.
    A window is never sent to Sentence Transformers if its tokenized input,
    including special tokens, exceeds the model's configured sequence length.
    """

    _models: dict[tuple[str, str], Any] = {}
    _models_lock = threading.Lock()

    def __init__(self, settings: EmbeddingSettings, *, model: Any | None = None):
        self.settings = settings
        self._provided_model = model
        self._dimension: int | None = None
        self._windowing_profile: dict[str, Any] | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def windowing_profile(self) -> dict[str, Any] | None:
        return dict(self._windowing_profile) if self._windowing_profile is not None else None

    @property
    def index_config(self) -> dict[str, Any]:
        config = self.settings.as_dict()
        if self._windowing_profile is not None:
            config["windowing"] = dict(self._windowing_profile)
        return config

    def _load_model(self) -> Any:
        if self._provided_model is not None:
            return self._provided_model
        key = (self.settings.model_name, self.settings.device)
        if key not in self._models:
            with self._models_lock:
                if key not in self._models:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise EmbeddingError(
                            "Sentence Transformers is not installed; install project requirements"
                        ) from exc
                    try:
                        self._models[key] = SentenceTransformer(
                            self.settings.model_name,
                            device=self.settings.device,
                        )
                    except Exception as exc:
                        raise EmbeddingError(
                            f"Could not load embedding model {self.settings.model_name!r} "
                            f"on {self.settings.device!r}: {exc}"
                        ) from exc
        return self._models[key]

    @staticmethod
    def _tokenize(tokenizer: Any, text: str, *, offsets: bool = False) -> tuple[list[int], list[tuple[int, int]] | None]:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=offsets,
                verbose=False,
            )
        except Exception as exc:
            purpose = "offset tokenization" if offsets else "token counting"
            raise EmbeddingError(f"Tokenizer {purpose} failed: {exc}") from exc
        try:
            token_ids = encoded["input_ids"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            token_ids = [int(token) for token in token_ids]
            raw_offsets = encoded.get("offset_mapping") if offsets else None
            if raw_offsets and isinstance(raw_offsets[0][0], (list, tuple)):
                raw_offsets = raw_offsets[0]
            normalized_offsets = None
            if offsets:
                if raw_offsets is None or len(raw_offsets) != len(token_ids):
                    raise ValueError("tokenizer did not return one offset per token")
                normalized_offsets = [(int(pair[0]), int(pair[1])) for pair in raw_offsets]
            return token_ids, normalized_offsets
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise EmbeddingError(f"Tokenizer returned invalid token data: {exc}") from exc

    def _profile(self, model: Any) -> tuple[Any, int, int, int]:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise EmbeddingError("Embedding model does not expose its tokenizer; safe truncation checks are unavailable")
        try:
            maximum = int(getattr(model, "max_seq_length"))
            special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        except Exception as exc:
            raise EmbeddingError(f"Could not determine the model token limit: {exc}") from exc
        usable = maximum - special_count
        if maximum <= 0 or special_count < 0 or usable <= 0:
            raise EmbeddingError("Embedding model has an invalid maximum sequence length or special-token budget")
        tokenizer_type = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        self._windowing_profile = {
            "strategy": WINDOWING_STRATEGY,
            "overlap_tokens": self.settings.window_overlap_tokens,
            "max_sequence_length": maximum,
            "special_token_count": special_count,
            "usable_content_tokens": usable,
            "tokenizer_class": tokenizer_type,
        }
        return tokenizer, maximum, special_count, usable

    def _validated_model_inputs(self, texts: list[str], *, role: str) -> tuple[Any, Any]:
        model = self._load_model()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            if self._provided_model is not None:
                # Test doubles have no tokenizer and do not truncate.
                method_name = "encode_document" if role == "document" else "encode_query"
                method = getattr(model, method_name, None) or getattr(model, "encode", None)
                if method is None:
                    raise EmbeddingError("Loaded model does not provide an encoding method")
                return model, method
            raise EmbeddingError("Embedding model does not expose its tokenizer; safe truncation checks are unavailable")
        _, maximum, special_count, _ = self._profile(model)
        for position, text in enumerate(texts):
            token_ids, _ = self._tokenize(tokenizer, text)
            actual_length = len(token_ids) + special_count
            if actual_length > maximum:
                raise EmbeddingError(
                    f"Embedding {role} input {position} has {actual_length} tokens including special tokens; "
                    f"model limit is {maximum}. Long source chunks must use token-aware windows."
                )
        method_name = "encode_document" if role == "document" else "encode_query"
        method = getattr(model, method_name, None) or getattr(model, "encode", None)
        if method is None:
            raise EmbeddingError("Loaded model does not provide an encoding method")
        return model, method

    def encode_documents(self, texts: list[str]):
        """Encode already short document inputs; over-limit text is rejected."""
        return self._encode(texts, role="document")

    def encode_queries(self, texts: list[str]):
        """Encode queries only when the complete query fits the model limit."""
        return self._encode(texts, role="query")

    def document_windows(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepare deterministic windows, retaining their original source mapping."""
        if not chunks:
            return []
        model = self._load_model()
        tokenizer, maximum, special_count, usable = self._profile(model)
        windows: list[dict[str, Any]] = []
        for chunk in sorted(chunks, key=lambda item: str(item.get("chunk_id", ""))):
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id:
                raise EmbeddingError("Every source chunk must have a chunk_id before windowing")
            code = str(chunk.get("code", ""))
            if not code.strip():
                continue
            prefix = "\n".join(filter(None, (
                f"File: {chunk.get('file_path', '')}",
                f"Type: {chunk.get('kind', '')}",
                f"Symbol: {chunk.get('symbol') or ''}",
            ))) + "\n"
            full_text = prefix + code
            all_ids, _ = self._tokenize(tokenizer, full_text)
            full_length = len(all_ids) + special_count
            if full_length <= maximum:
                span_rows = [(0, len(code), 0, len(all_ids))]
            else:
                code_ids, offsets = self._tokenize(tokenizer, code, offsets=True)
                usable_offsets = [pair for pair in (offsets or []) if pair[1] > pair[0]]
                if len(usable_offsets) != len(code_ids) or not code_ids:
                    raise EmbeddingError(
                        f"Tokenizer could not map source offsets for oversized chunk {chunk_id!r}"
                    )
                prefix_ids, _ = self._tokenize(tokenizer, prefix)
                code_budget = usable - len(prefix_ids)
                if code_budget < 1:
                    raise EmbeddingError(
                        f"Chunk metadata prefix uses the model's {usable}-token content budget"
                    )
                overlap = self.settings.window_overlap_tokens
                if overlap >= code_budget:
                    raise EmbeddingError(
                        f"Window overlap ({overlap}) must be smaller than the {code_budget}-token code window"
                    )
                span_rows = []
                token_start = 0
                while token_start < len(code_ids):
                    token_end = min(len(code_ids), token_start + code_budget)

                    def span(end_index: int) -> tuple[int, int, str, int]:
                        char_start = usable_offsets[token_start][0]
                        char_end = usable_offsets[end_index - 1][1]
                        text = prefix + code[char_start:char_end]
                        count = len(self._tokenize(tokenizer, text)[0]) + special_count
                        return char_start, char_end, text, count

                    char_start, char_end, text, actual_length = span(token_end)
                    while actual_length > maximum and token_end > token_start + 1:
                        token_end -= 1
                        char_start, char_end, text, actual_length = span(token_end)
                    if actual_length > maximum:
                        raise EmbeddingError(
                            f"Tokenizer window for chunk {chunk_id!r} cannot fit the model sequence limit"
                        )
                    span_rows.append((char_start, char_end, token_start, token_end, text))
                    if token_end >= len(code_ids):
                        break
                    next_start = token_end - overlap
                    if next_start <= token_start:
                        raise EmbeddingError("Windowing failed to advance across the source chunk")
                    token_start = next_start

            count = len(span_rows)
            for index, row in enumerate(span_rows, 1):
                if len(row) == 4:
                    char_start, char_end, token_start, token_end = row
                    embedding_text = full_text
                else:
                    char_start, char_end, token_start, token_end, embedding_text = row
                if char_end <= char_start:
                    raise EmbeddingError(f"Tokenizer produced an empty source window for chunk {chunk_id!r}")
                local_line_start = code.count("\n", 0, char_start) + 1
                local_line_end = code.count("\n", 0, max(char_start, char_end - 1)) + 1
                line_start = int(chunk.get("line_start", 1)) + local_line_start - 1
                line_end = min(
                    int(chunk.get("line_end", line_start + local_line_end - local_line_start)),
                    int(chunk.get("line_start", 1)) + local_line_end - 1,
                )
                window_id = hashlib.sha256(
                    f"{chunk_id}\0{char_start}\0{char_end}\0{code[char_start:char_end]}".encode("utf-8")
                ).hexdigest()[:24]
                windows.append({
                    "chunk_id": chunk_id,
                    "parent_chunk_id": chunk_id,
                    "file_path": chunk.get("file_path"),
                    "kind": chunk.get("kind"),
                    "symbol": chunk.get("symbol"),
                    "part": chunk.get("part"),
                    "line_start": line_start,
                    "line_end": max(line_start, line_end),
                    "window_id": window_id,
                    "window_index": index,
                    "window_count": count,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": token_start,
                    "token_end": token_end,
                    "embedding_text": embedding_text,
                })
        return windows

    def encode_chunk_documents(self, chunks: list[dict[str, Any]]):
        """Window source chunks as needed and return (vectors, source mappings)."""
        import numpy as np

        windows = self.document_windows(chunks)
        if not windows:
            return np.empty((0, 0), dtype=np.float32), windows
        vectors = self.encode_document_windows(windows)
        return vectors, windows

    def encode_document_windows(self, windows: list[dict[str, Any]]):
        """Encode previously validated windows without changing their row order."""
        return self._encode([window["embedding_text"] for window in windows], role="document")

    def _encode(self, texts: list[str], *, role: str):
        import numpy as np

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding input must contain non-empty strings")

        model, method = self._validated_model_inputs(texts, role=role)
        try:
            encoded = method(
                texts,
                batch_size=self.settings.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding {role} text failed: {exc}") from exc

        vectors = np.asarray(encoded, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] <= 0:
            raise EmbeddingError("Embedding model returned an invalid matrix shape")
        if not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding model returned non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise EmbeddingError("Embedding model returned a zero-length vector")
        vectors = np.ascontiguousarray(vectors / norms[:, None], dtype=np.float32)
        if self._dimension is not None and vectors.shape[1] != self._dimension:
            raise EmbeddingError(
                f"Embedding dimension changed from {self._dimension} to {vectors.shape[1]}"
            )
        self._dimension = int(vectors.shape[1])
        return vectors
